# 2차 시도: 식당이름 + 주소로 검색 -> 실패하면, 식당이름 + 성동구로 검색

import pandas as pd
import os
import re
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = None
recovered_results = []

def setup_driver():
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("lang=ko_KR")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(2)

# 주소 정제 함수 (1차 검색용)
def clean_address_for_search(addr):
    if pd.isna(addr): return ""
    addr = str(addr)
    addr = re.sub(r'\(.*?\)', '', addr)
    addr = re.sub(r'[,.]', ' ', addr)
    return addr.strip()

#  지역명(구/시/군) 추출 함수 (2차 검색용)
def extract_region_keyword(addr):
    if pd.isna(addr): return ""
    tokens = str(addr).split()
    
    # 1순위: '구'로 끝나는 단어 우선 (예: 성동구, 팔달구)
    for token in tokens:
        if token.endswith("구"):
            return token
            
    # 2순위: '군'으로 끝나는 단어 (예: 가평군)
    for token in tokens:
        if token.endswith("군"):
            return token
            
    # 3순위: '시'로 끝나는 단어 (단, 특별시/광역시는 제외)
    # 예: '수원시'는 OK, '서울특별시'는 NO
    for token in tokens:
        if token.endswith("시") and "특별시" not in token and "광역시" not in token:
            return token
            
    # 4순위: 위에서 안 걸렸다면(예: 서울특별시 ...), 보통 두 번째 단어가 상세 지역임
    if len(tokens) > 1:
        return tokens[1]
    
    return ""

# 메인 함수
def main():
    global driver, recovered_results

    setup_driver()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')
        
        list_path = os.path.join(data_dir, 'kakao_store_list.csv')
        origin_path = os.path.join(data_dir, '최종 데이터셋.csv')
        output_path = os.path.join(data_dir, 'kakao_ratings_final.xlsx')
    except Exception as e:
        print(f"경로 설정 오류: {e}")
        return

    # 실패한 리스트 추출
    print("데이터 로드 중...")
    try:
        df_list = pd.read_csv(list_path)
        failed_ids = df_list[df_list['카카오맵식당명'].isin(['못찾음', '정보부족', '재검색실패', '오류'])]['가맹점구분번호'].astype(str).tolist()
        
        # 이미 성공한 것 제외
        if os.path.exists(output_path):
            try:
                df_exist = pd.read_excel(output_path)
                exist_ids = df_exist['가맹점구분번호'].astype(str).tolist()
                failed_ids = list(set(failed_ids) - set(exist_ids))
            except: pass

        if not failed_ids:
            print("재검색할 대상이 없습니다!")
            return

        df_origin = pd.read_csv(origin_path)
        df_origin['가맹점구분번호'] = df_origin['가맹점구분번호'].astype(str)
        
        target_df = df_origin[df_origin['가맹점구분번호'].isin(failed_ids)].copy()
        target_df = target_df.drop_duplicates(subset=['가맹점구분번호'])
        
        print(f"총 {len(target_df)}개의 실패 가맹점을 2단계로 재검색합니다.")

    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return

    # 재검색 시작
    driver.get('https://map.kakao.com/')
    sleep(1)

    total = len(target_df)
    
    for i, (idx, row) in enumerate(target_df.iterrows()):
        
        if i % 20 == 0 and i != 0:
            append_to_excel(output_path)
            print("   [중간 데이터 저장 완료]")

        f_id = row['가맹점구분번호']
        f_name = str(row['가맹점명']).replace('*', '').strip()
        f_addr_raw = row['가맹점주소']
        
        # 1차 검색어: [상세주소] [이름]
        clean_addr = clean_address_for_search(f_addr_raw)
        search_query_1 = f"{clean_addr} {f_name}"
        
        print(f"\n[{i+1}/{total}] 1차 검색: {search_query_1}")
        
        found = False
        try:
            # 1차 시도 실행
            if perform_search(search_query_1):
                found = find_and_crawl_deep(f_id, f_name)
            
            # 2차 시도 (1차 실패 시) - 지역명(구/시) + 이름
            if not found:
                region = extract_region_keyword(f_addr_raw)
                if region:
                    search_query_2 = f"{region} {f_name}"
                    print(f"   ↪ 실패.. 2차 검색(지역+이름): {search_query_2}")
                    
                    if perform_search(search_query_2):
                        found = find_and_crawl_deep(f_id, f_name)

            if not found:
                 print("   >>> 최종 실패 (목록에 없음)")
        
        except Exception as e:
            print(f"   오류 발생: {e}")
            try: driver.switch_to.window(driver.window_handles[0])
            except: pass

    # 최종 저장
    append_to_excel(output_path)
    print("\n 모든 재검색이 완료되었습니다.")
    driver.quit()

# 검색 실행 함수
def perform_search(query):
    global driver
    try:
        search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')
        search_area.clear()
        search_area.send_keys(query)
        driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
        sleep(1.5)
        return True
    except: return False

# 장소 더보기 & 페이지 탐색 함수
def find_and_crawl_deep(f_id, target_name):
    global driver, recovered_results

    # 장소 더보기 클릭
    try:
        more_btn = driver.find_element(By.ID, "info.search.place.more")
        if more_btn.is_displayed():
            driver.execute_script("arguments[0].click();", more_btn)
            sleep(1)
    except: pass

    # 1~5페이지 탐색
    for page in range(1, 6):
        try:
            if page > 1:
                page_btn = driver.find_element(By.ID, f"info.search.page.no{page}")
                if "disabled" in page_btn.get_attribute("class"): break
                driver.execute_script("arguments[0].click();", page_btn)
                sleep(1)

            places = driver.find_elements(By.CSS_SELECTOR, '.placelist > .PlaceItem')
            if not places: break

            for place in places:
                try:
                    p_name = place.find_element(By.CSS_SELECTOR, '.head_item > .tit_name > .link_name').text
                    
                    # 이름 비교 (공백 제거 후 확인)
                    if target_name.replace(" ", "") in p_name.replace(" ", ""):
                        print(f"      찾음! ({p_name}) -> 수집 시작")
                        
                        detail_btn = place.find_element(By.CSS_SELECTOR, "a.moreview")
                        driver.execute_script("arguments[0].click();", detail_btn)
                        
                        driver.switch_to.window(driver.window_handles[-1])
                        sleep(1.5)
                        
                        click_review_tab()
                        data = extract_details_data(f_id, p_name) 
                        recovered_results.append(data)
                        
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        return True
                except: continue
        except: break
            
    return False

# 상세 정보 추출
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

    print(f"         -> 결과: {overall_rating} | {category_str[:15]}...")
    return [f_id, store_name, overall_rating, category_str]

# 후기 탭 클릭
def click_review_tab():
    try:
        review_tab = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[href="#review"].link_tab'))
        )
        driver.execute_script("arguments[0].click();", review_tab)
        sleep(1)
    except: pass

# 엑셀 이어쓰기
def append_to_excel(path):
    global recovered_results
    if not recovered_results: return
    
    new_df = pd.DataFrame(recovered_results, columns=['가맹점구분번호', '식당명', '종합별점', '카테고리평가'])
    
    if os.path.exists(path):
        try:
            existing_df = pd.read_excel(path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(path, index=False)
        except:
            new_df.to_csv(path.replace('.xlsx', '_recovered.csv'), index=False, encoding='utf-8-sig')
    else:
        new_df.to_excel(path, index=False)
    
    recovered_results = []

if __name__ == "__main__":
    main()