# 1차 크롤링 시도: 주소 검색되는 가맹점들(kakao_store_list)만 추출

import pandas as pd
import os
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = None
final_results = []

def setup_driver():
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("lang=ko_KR") 
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(3)

# 메인 함수
def main():
    global driver, final_results

    setup_driver()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        input_path = os.path.join(data_dir, 'kakao_store_list.csv')
        output_path = os.path.join(data_dir, 'kakao_ratings_final.xlsx')
        
        print(f"읽어올 파일: {input_path}")
        df_list = pd.read_csv(input_path)
    except FileNotFoundError:
        print("오류: 'kakao_store_list.csv' 파일이 없습니다.")
        return

    # '못찾음'이 아닌 데이터만 필터링
    target_df = df_list[df_list['카카오맵식당명'] != '못찾음'].copy()
    target_df = target_df.reset_index(drop=True)
    
    total_count = len(target_df)
    print(f"총 {total_count}개의 식당을 '이름 검색' 방식으로 크롤링합니다.")

    driver.get('https://map.kakao.com/')
    sleep(2)

    # 크롤링 루프
    for i, row in target_df.iterrows():
        if i % 50 == 0 and i != 0:
            save_to_excel(output_path)
            print(f"   [자동저장] {i}/{total_count} 완료")
        
        f_id = row['가맹점구분번호']
        store_name = row['카카오맵식당명'] # 검색할 이름
        
        print(f"[{i+1}/{total_count}] 검색: {store_name}")
        
        try:
            # 검색어 입력
            search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')
            search_area.clear()
            search_area.send_keys(store_name)
            driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
            sleep(1.5) # 검색 결과 로딩 대기

            # 첫 번째 결과 클릭
            # 검색 결과 리스트 가져오기
            places = driver.find_elements(By.CSS_SELECTOR, '.placelist > .PlaceItem')
            
            if len(places) > 0:
                # 첫 번째 장소의 '상세보기' 버튼 클릭
                first_place = places[0]
                detail_btn = first_place.find_element(By.CSS_SELECTOR, "a.moreview")
                detail_btn.send_keys(Keys.ENTER)
                
                # 새 탭으로 전환
                driver.switch_to.window(driver.window_handles[-1])
                sleep(1.5) # 상세페이지 로딩 대기
                
                # 후기 탭 클릭 및 수집
                click_review_tab()
                extract_details(f_id, store_name)
                
                # 탭 닫기 및 복귀
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            else:
                print(f"   >>> 검색 결과 없음 (이상함, 1단계에선 있었음)")
                final_results.append([f_id, store_name, "재검색실패", ""])

        except Exception as e:
            print(f"오류: {e}")
            final_results.append([f_id, store_name, "오류", str(e)])
            # 혹시 탭이 꼬였을 경우를 대비한 복구 코드
            try:
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
            except: pass
            continue

    # 최종 저장
    save_to_excel(output_path)
    print(f"\n 완료! '{output_path}' 저장 끝.")
    driver.quit()

# 후기 탭 클릭 함수
def click_review_tab():
    global driver
    try:
        # 후기 탭이 클릭 가능해질 때까지 대기
        review_tab = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[href="#review"].link_tab'))
        )
        driver.execute_script("arguments[0].click();", review_tab)
        sleep(1) # 데이터 로딩 대기
    except:
        pass # 탭이 없으면 넘어감

# 상세 정보 추출 함수
def extract_details(f_id, store_name):
    global driver, final_results
    
    # 별점
    overall_rating = "0.0"
    try:
        rating_elem = driver.find_element(By.CSS_SELECTOR, ".starred_grade .num_star")
        overall_rating = rating_elem.text
    except:
        overall_rating = "0.0"
    
    # 카테고리
    category_str = ""
    try:
        category_data = []
        # 리스트 로딩 대기
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
                if nm and cnt:
                    category_data.append(f"{nm}:{cnt}")
            except: continue
        
        if category_data:
            category_str = ", ".join(category_data)
        else:
            category_str = "카테고리없음"
    except:
        category_str = "수집실패"

    final_results.append([f_id, store_name, overall_rating, category_str])
    
    # 로그 출력 (간략히)
    disp = category_str if len(category_str) < 30 else category_str[:30] + "..."
    print(f" 수집: {overall_rating} | {disp}")

# 엑셀 저장
def save_to_excel(path):
    df = pd.DataFrame(final_results, columns=['가맹점구분번호', '식당명', '종합별점', '카테고리평가'])
    df.to_excel(path, index=False)

if __name__ == "__main__":
    main()