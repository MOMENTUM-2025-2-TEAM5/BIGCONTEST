# 1차 주소 검색하여 나타나는 식당명을 엑셀로 저장하는 코드

import pandas as pd
import os
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = None
found_stores_list = [] # 결과 저장 리스트

def setup_driver():
    global driver
    options = webdriver.ChromeOptions()
    options.add_argument("lang=ko_KR")
    # 속도를 위해 화면 없이 실행 (Headless 모드)
    options.add_argument("headless") 
    # 일부 환경에서 Headless 감지를 피하기 위한 설정
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.implicitly_wait(3)

# 메인 함수
def main():
    global driver, found_stores_list

    setup_driver()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data') # data 폴더 경로 지정
        
        # 입력 파일 경로
        csv_path = os.path.join(data_dir, '최종 데이터셋.csv')
        
        # 출력 파일 경로
        output_path = os.path.join(data_dir, 'kakao_store_list.csv')
        
        print(f"읽어올 파일: {csv_path}")
        print(f"저장할 파일: {output_path}")

        df_data = pd.read_csv(csv_path)

    except Exception as e:
        print(f"오류: 파일을 찾거나 경로를 설정하는 중 문제가 발생했습니다.\n{e}")
        return

    required_cols = ['가맹점구분번호', '가맹점명', '가맹점주소']
    if not all(col in df_data.columns for col in required_cols):
        print(f"오류: 필수 컬럼({required_cols})이 없습니다.")
        return
    
    original_count = len(df_data)
    # 가맹점구분번호 기준 중복 제거
    df_data = df_data.drop_duplicates(subset=['가맹점구분번호'], keep='first')
    # 인덱스 리셋 (0, 1, 2... 순서로 깔끔하게)
    df_data = df_data.reset_index(drop=True)
    dedup_count = len(df_data)
    
    print(f"원본 데이터 {original_count}개 -> 중복 제거 후 {dedup_count}개")
    print("크롤링 시작...\n")
    
    driver.get('https://map.kakao.com/')

    # 크롤링 루프
    for i, row in df_data.iterrows():
        if i % 50 == 0 and i != 0:
            sleep(2)
            # 중간 저장 (50개마다 엑셀 파일 업데이트)
            save_to_csv(output_path)
            print(f"   [자동저장] {i}번째까지 데이터 저장 완료.")

        franchise_id = row['가맹점구분번호']
        franchise_name = row['가맹점명']
        address = row['가맹점주소']
        
        # 진행 상황 표시
        progress_str = f"{i+1}/{dedup_count}"
        
        if pd.isna(address) or pd.isna(franchise_name):
            print(f"[{progress_str}] 정보 부족 (ID: {franchise_id}) -> 저장(실패)")
            found_stores_list.append([franchise_id, franchise_name, "정보부족", ""])
            continue
        
        print(f"[{progress_str}] 검색: {address} (매장: {franchise_name})")
        
        # 검색 및 URL 찾기 시도
        found = search_and_find_url(franchise_id, franchise_name, address)
        
        # 못 찾았을 경우 '못찾음'으로 저장
        if not found:
            print(f"   >>> 검색 실패 -> '못찾음' 저장")
            found_stores_list.append([franchise_id, franchise_name, "못찾음", ""])

    # 최종 저장
    save_to_csv(output_path)
    print(f"\n완료! 모든 데이터가 '{output_path}'에 저장되었습니다.")
    driver.quit()

# CSV 저장 헬퍼 함수
def save_to_csv(path):
    df_list = pd.DataFrame(found_stores_list, columns=['가맹점구분번호', 'CSV가맹점명', '카카오맵식당명', '카카오맵URL'])
    # utf-8-sig로 저장해야 엑셀에서 한글이 안 깨짐
    df_list.to_csv(path, index=False, encoding='utf-8-sig')

# 검색 로직
def search_and_find_url(franchise_id, franchise_name, address):
    global driver
    try:
        search_area = driver.find_element(By.XPATH, '//*[@id="search.keyword.query"]')
        search_area.clear()
        search_area.send_keys(address)
        driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
        sleep(1)
    except Exception:
        return False

    for page in range(1, 6):
        try:
            places = driver.find_elements(By.CSS_SELECTOR, '.placelist > .PlaceItem')
            if not places:
                return False
        except Exception:
            return False

        # 현재 페이지에서 매칭 시도
        if find_matching_url_in_page(franchise_id, franchise_name, places):
            return True

        # 다음 페이지 이동
        try:
            page_btn = driver.find_element(By.ID, f'info.search.page.no{page + 1}')
            page_btn.send_keys(Keys.ENTER)
            sleep(1)
        except:
            return False # 다음 페이지 없음

    return False

# 매칭 확인 로직
def find_matching_url_in_page(franchise_id, franchise_name, place_elements):
    global found_stores_list
    
    csv_name_clean = str(franchise_name).replace('*', '').strip()
    if not csv_name_clean:
        return False

    for place in place_elements:
        try:
            place_name = place.find_element(By.CSS_SELECTOR, '.head_item > .tit_name > .link_name').text
            
            if csv_name_clean in place_name:
                detail_btn = place.find_element(By.CSS_SELECTOR, "a.moreview")
                href = detail_btn.get_attribute('href')
                
                if href:
                    full_url = "https://map.kakao.com" + href
                    print(f"   >>> 성공: {place_name}")
                    found_stores_list.append([franchise_id, franchise_name, place_name, full_url])
                    return True
        except Exception:
            continue
            
    return False

if __name__ == "__main__":
    main()