# [4단계: final_crawl_strategy.py] - 주소 검색 + 페이지네이션 정밀 탐색
# 3차 시도 (도대체 언제 끝남)

import pandas as pd
import os
import re
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- 전역 변수 ---
driver = None
final_found_results = [] # 찾은 결과 저장

# --- 드라이버 설정 ---
def setup_driver():
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("lang=ko_KR")
    # options.add_argument("headless") # 화면 보면서 확인 (주석 해제 시 안보임)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(2)

# --- 주소 정제 (건물 단위까지만 남기기) ---
def clean_address_strict(addr):
    if pd.isna(addr): return ""
    addr = str(addr)
    # 괄호 제거
    addr = re.sub(r'\(.*?\)', '', addr)
    # 층, 호, 지하 제거 (검색 범위를 건물 전체로 넓힘)
    addr = re.sub(r'\d+층|\d+호|지하\d+|B\d+', '', addr)
    addr = re.sub(r'[,.]', ' ', addr)
    return addr.strip()

# --- 메인 함수 ---
def main():
    global driver, final_found_results

    setup_driver()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        # 입력: 미검색 가맹점 목록
        input_path = os.path.join(data_dir, '미검색_가맹점_목록.xlsx')
        # 출력: 추가 수집된 데이터
        output_path = os.path.join(data_dir, 'kakao_ratings_supplement.xlsx')
    except Exception as e:
        print(f"경로 오류: {e}")
        return

    print("데이터 로드 중...")
    try:
        df_target = pd.read_excel(input_path)
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return

    print(f"총 {len(df_target)}개의 '숨겨진 맛집' 발굴을 시작합니다.")
    
    driver.get('https://map.kakao.com/')
    sleep(1)

    total = len(df_target)
    
    for i, row in df_target.iterrows():
        # 중간 저장
        if i % 10 == 0 and i != 0:
            save_to_excel(output_path)
            print("   [중간 저장 완료]")

        f_id = row['가맹점구분번호']
        f_name_raw = str(row['가맹점명'])
        # 이름에서 별표 제거 (예: 밥플*********** -> 밥플)
        f_name_clean = f_name_raw.replace('*', '').strip()
        
        # 주소 정제
        f_addr_raw = row['가맹점주소']
        search_addr = clean_address_strict(f_addr_raw)
        
        print(f"\n[{i+1}/{total}] 목표: {f_name_clean} (주소: {search_addr})")
        
        found = False
        try:
            # 1. 주소로 검색
            search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')
            search_area.clear()
            search_area.send_keys(search_addr)
            driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
            sleep(1.5)

            # 2. 장소 더보기 + 1~5페이지 뒤지기
            found = find_and_crawl_deep_by_address(f_id, f_name_clean)
            
            if not found:
                print("   >>> ❌ 해당 주소 내에서 이름 일치 가게 없음")

        except Exception as e:
            print(f"   ❌ 오류: {e}")
            try: driver.switch_to.window(driver.window_handles[0])
            except: pass

    # 최종 저장
    save_to_excel(output_path)
    print("\n🎉 모든 작업 완료! 수고하셨습니다.")
    driver.quit()


# --- 📌 [핵심] 주소 검색 결과 내 정밀 탐색 함수 ---
def find_and_crawl_deep_by_address(f_id, target_name_prefix):
    global driver, final_found_results

    # 1. "장소 더보기" 버튼 클릭 시도
    # (결과가 5개 미만이면 더보기가 없을 수도 있음 -> pass)
    try:
        more_btn = driver.find_element(By.ID, "info.search.place.more")
        if more_btn.is_displayed():
            driver.execute_script("arguments[0].click();", more_btn)
            sleep(1)
            # print("   (장소 더보기 클릭)")
    except:
        pass

    # 2. 1페이지 ~ 5페이지 순회
    for page in range(1, 6):
        try:
            # 페이지 번호 클릭
            if page > 1:
                page_btn = driver.find_element(By.ID, f"info.search.page.no{page}")
                
                # 버튼이 비활성화(disabled) 상태면 더 이상 페이지가 없는 것
                if "disabled" in page_btn.get_attribute("class"):
                    break 
                
                driver.execute_script("arguments[0].click();", page_btn)
                sleep(1) # 목록 로딩 대기

            # 현재 페이지의 장소 목록 가져오기
            places = driver.find_elements(By.CSS_SELECTOR, '.placelist > .PlaceItem')
            
            if not places: break

            # 목록 하나씩 확인
            for place in places:
                try:
                    p_name = place.find_element(By.CSS_SELECTOR, '.head_item > .tit_name > .link_name').text
                    
                    # 📌 이름 매칭 로직
                    # 카카오맵 식당 이름(p_name)에 우리 엑셀의 '별표 뗀 이름'이 포함되는지 확인
                    # 예: 엑셀(돈벼) vs 카카오(돈벼락 맛집) -> 매칭 성공
                    if target_name_prefix.replace(" ", "") in p_name.replace(" ", ""):
                        print(f"   ✅ 매칭 성공! ({p_name}) [페이지:{page}] -> 수집")
                        
                        # 상세보기 클릭
                        detail_btn = place.find_element(By.CSS_SELECTOR, "a.moreview")
                        driver.execute_script("arguments[0].click();", detail_btn)
                        
                        driver.switch_to.window(driver.window_handles[-1])
                        sleep(1.5)
                        
                        # 정보 수집
                        click_review_tab()
                        data = extract_details_data(f_id, p_name)
                        final_found_results.append(data)
                        
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        return True # 찾았으니 종료
                except:
                    continue
        except NoSuchElementException:
            break
        except Exception as e:
            print(f"   (페이지 {page} 탐색 중 에러: {e})")
            continue
            
    return False

# --- 후기 탭 클릭 ---
def click_review_tab():
    try:
        review_tab = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[href="#review"].link_tab'))
        )
        driver.execute_script("arguments[0].click();", review_tab)
        sleep(1)
    except: pass

# --- 데이터 수집 ---
def extract_details_data(f_id, store_name):
    global driver
    
    # 별점
    overall_rating = "0.0"
    try:
        overall_rating = driver.find_element(By.CSS_SELECTOR, ".starred_grade .num_star").text
    except: pass
    
    # 카테고리
    category_str = ""
    try:
        category_data = []
        try:
            WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".wrap_point .list_point"))
            )
        except: pass

        items = driver.find_elements(By.CSS_SELECTOR, ".wrap_point .list_point > li")
        for item in items:
            if item.get_attribute("aria-hidden") == "true": continue
            try:
                nm = item.find_element(By.CSS_SELECTOR, ".txt_point").text
                cnt = item.find_element(By.CSS_SELECTOR, ".rate_point").text
                if nm and cnt: category_data.append(f"{nm}:{cnt}")
            except: continue
        category_str = ", ".join(category_data) if category_data else "카테고리없음"
    except: category_str = "수집실패"

    print(f"      -> {overall_rating} | {category_str[:15]}...")
    return [f_id, store_name, overall_rating, category_str]

# --- 엑셀 저장 ---
def save_to_excel(path):
    global final_found_results
    if not final_found_results: return
    
    df = pd.DataFrame(final_found_results, columns=['가맹점구분번호', '식당명', '종합별점', '카테고리평가'])
    # 기존 파일 있으면 이어쓰기 모드처럼 보이기 위해 읽어서 합침
    if os.path.exists(path):
        try:
            df_ex = pd.read_excel(path)
            df = pd.concat([df_ex, df], ignore_index=True).drop_duplicates()
        except: pass
        
    df.to_excel(path, index=False)

if __name__ == "__main__":
    main()