# 4차 시도

import pandas as pd
import os
import re
from datetime import datetime
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

driver = None
success_list = [] # 성공 목록
fail_list = []    # 실패 목록

def setup_driver():
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("lang=ko_KR")
    # options.add_argument("headless") # 화면 보려면 주석 처리 유지
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(2)

# 식당명 정제 함수
def clean_text_for_search(text):
    if pd.isna(text): return ""
    text = str(text)
    text = text.replace('*', '') # 별표 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text) # 한글, 영어, 숫자만 남기고 나머지(점, 쉼표, 특수문자 등) 제거
    return text.strip()

# 주소 정제 함수
def clean_address(addr):
    if pd.isna(addr): return ""
    addr = str(addr)
    # 괄호, 층, 호, 지하 제거 (건물 전체 검색 유도)
    addr = re.sub(r'\(.*?\)', '', addr)
    addr = re.sub(r'\d+층|\d+호|지하\d+|B\d+', '', addr)
    # 점, 쉼표 제거
    addr = re.sub(r'[,.]', ' ', addr)
    return addr.strip()

# 메인 함수
def main():
    global driver, success_list, fail_list

    setup_driver()

    # 파일 경로 설정
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')
        input_path = os.path.join(data_dir, '최종_미검색_가맹점.xlsx')
    except Exception as e:
        print(f"경로 오류: {e}")
        return

    # 실시간 날짜 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    success_path = os.path.join(data_dir, f'검색성공_{timestamp}.xlsx')
    fail_path = os.path.join(data_dir, f'검색실패_{timestamp}.xlsx')

    # 데이터 로드
    try:
        df_target = pd.read_excel(input_path)
        print(f"총 {len(df_target)}개 가맹점 정밀 탐색 시작")
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return

    driver.get('https://map.kakao.com/')
    sleep(1)

    total = len(df_target)
    
    # 크롤링 루프
    for i, row in df_target.iterrows():
        # 중간 저장 (10개마다)
        if i % 10 == 0 and i != 0:
            save_files(success_path, fail_path)
            print("   [중간 저장 완료]")

        f_id = row['가맹점구분번호']
        f_name_raw = str(row['가맹점명']) 
        f_addr_raw = row['가맹점주소'] 
        
        # 목표 이름 결정 (생략)
        target_name_raw = ""
        if '식당명' in row and pd.notna(row['식당명']) and str(row['식당명']) != '현재검색X':
            target_name_raw = str(row['식당명'])
        else:
            target_name_raw = f_name_raw
        
        # 이름과 주소 정제
        target_name_clean = clean_text_for_search(target_name_raw)
        search_addr = clean_address(f_addr_raw)
        
        print(f"\n[{i+1}/{total}] 목표: {target_name_clean} (원본: {target_name_raw})")
        
        found = False
        try:
            # 1차 시도: 주소 검색
            print(f"1차(주소): {search_addr}")
            perform_search(search_addr)
            
            # 페이지 끝까지 넘기며 찾기
            found_data = deep_pagination_scan(f_id, f_name_raw, f_addr_raw, target_name_clean)
            
            if found_data:
                success_list.append(found_data)
                found = True
                print("1차에서 발견 성공")

            # 2차 시도: "식당이름 + 성동구" 검색
            if not found:
                search_query_2 = f"{target_name_clean} 성동구"
                print(f"   2차(이름+지역): {search_query_2}")
                perform_search(search_query_2)
                
                found_data = deep_pagination_scan(f_id, f_name_raw, f_addr_raw, target_name_clean)
                
                if found_data:
                    success_list.append(found_data)
                    found = True
                    print("2차에서 발견 성공")
            
            # 최종 실패 처리 -> 검색실패로 저장
            if not found:
                print("최종 실패 (목록에 없음)")
                fail_list.append([f_id, f_name_raw, f_addr_raw, "검색실패"])

        except Exception as e:
            print(f"오류: {e}")
            fail_list.append([f_id, f_name_raw, f_addr_raw, f"시스템오류: {str(e)}"])
            try: driver.switch_to.window(driver.window_handles[0])
            except: pass

    # 최종 저장
    save_files(success_path, fail_path)
    print("\n 작업 완료")
    driver.quit()


# --- 검색어 입력 함수 ---
def perform_search(query):
    global driver
    try:
        search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')
        search_area.clear()
        search_area.send_keys(query)
        driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
        sleep(1.5)
    except: pass

# 페이지네이션 탐색 (더보기 -> 1~5 -> 다음 -> 6~10...)
def deep_pagination_scan(f_id, f_name_raw, f_addr_raw, target_name_clean):
    global driver

    # 1. '장소 더보기' 클릭
    try:
        more_btn = driver.find_element(By.ID, "info.search.place.more")
        if more_btn.is_displayed():
            driver.execute_script("arguments[0].click();", more_btn)
            sleep(1)
    except: pass

    # 2. 페이지 그룹 순회 (최대 5그룹 = 25페이지까지 확인)
    page_group = 0
    max_groups = 5

    while page_group < max_groups:
        # 현재 그룹의 1~5 페이지 확인
        for page_idx in range(1, 6):
            try:
                # 페이지 번호 버튼 찾기
                page_btn_id = f"info.search.page.no{page_idx}"
                try:
                    page_btn = driver.find_element(By.ID, page_btn_id)
                    driver.execute_script("arguments[0].click();", page_btn)
                    sleep(0.8)
                except NoSuchElementException:
                    return None # 버튼 없으면 끝

                # 목록 스캔
                places = driver.find_elements(By.CSS_SELECTOR, '.placelist > .PlaceItem')
                if not places: return None

                for place in places:
                    try:
                        p_name = place.find_element(By.CSS_SELECTOR, '.head_item > .tit_name > .link_name').text
                        
                        # [비교 로직] 검색된 이름도 특수문자 다 떼고 비교!
                        p_name_clean = clean_text_for_search(p_name)
                        
                        # 예: target="앂" in found="앂아로마" -> True
                        if target_name_clean in p_name_clean:
                            print(f"매칭 확인: {p_name} (페이지:{page_group*5 + page_idx})")
                            
                            # 상세보기 -> 수집
                            detail_btn = place.find_element(By.CSS_SELECTOR, "a.moreview")
                            driver.execute_script("arguments[0].click();", detail_btn)
                            
                            driver.switch_to.window(driver.window_handles[-1])
                            sleep(1.5)
                            
                            click_review_tab()
                            # 데이터 추출
                            rating, cate = extract_details()
                            
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                            
                            # 결과 반환
                            return [f_id, f_name_raw, f_addr_raw, rating, cate]
                    except: continue
                # 스캔 끝

            except Exception:
                continue
        
        # 3. '다음' 버튼 확인
        try:
            next_btn = driver.find_element(By.ID, "info.search.page.next")
            if "disabled" in next_btn.get_attribute("class"):
                break # 다음 페이지 없음
            
            driver.execute_script("arguments[0].click();", next_btn)
            sleep(1)
            page_group += 1
        except:
            break

    return None # 못 찾음

# 데이터 수집 (별점/카테고리)
def extract_details():
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

    print(f"         -> 수집: {overall_rating} | {category_str[:15]}...")
    return overall_rating, category_str

# 후기 탭 클릭
def click_review_tab():
    try:
        review_tab = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[href="#review"].link_tab'))
        )
        driver.execute_script("arguments[0].click();", review_tab)
        sleep(1)
    except: pass

# 파일 저장
def save_files(success_path, fail_path):
    global success_list, fail_list
    
    # 성공 파일 저장
    if success_list:
        df_success = pd.DataFrame(success_list, columns=['가맹점구분번호', '가맹점명', '주소', '별점', '카테고리평가'])
        df_success.to_excel(success_path, index=False)
    
    # 실패 파일 저장
    if fail_list:
        df_fail = pd.DataFrame(fail_list, columns=['가맹점구분번호', '가맹점명', '주소', '비고'])
        df_fail.to_excel(fail_path, index=False)

if __name__ == "__main__":
    main()