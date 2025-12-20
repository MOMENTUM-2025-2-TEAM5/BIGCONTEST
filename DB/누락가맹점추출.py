# 1차, 2차에서 통합한 후, 아직 검색되지 않은 가맹점 엑셀로 저장하는 코드

import pandas as pd
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')

        # 전체 원본 파일
        origin_path = os.path.join(data_dir, '최종 데이터셋.csv') 
        # 현재까지 크롤링 완료된 통합 파일
        crawled_path = os.path.join(data_dir, '카카오크롤링_최종_통합본.xlsx')
        # 결과 저장 파일
        output_path = os.path.join(data_dir, '미검색_가맹점_목록.xlsx')
        
    except Exception as e:
        print(f"경로 설정 오류: {e}")
        return

    # 원본 데이터 로드
    try:
        df_origin = pd.read_csv(origin_path)
    except:
        df_origin = pd.read_excel(origin_path)

    # 크롤링 완료 데이터 로드
    try:
        df_crawled = pd.read_excel(crawled_path)
    except:
        df_crawled = pd.read_csv(crawled_path)

    # 데이터 비교
    
    # 비교를 위해 ID 컬럼을 문자열(String)로 통일 (숫자/문자 혼용 방지)
    df_origin['가맹점구분번호'] = df_origin['가맹점구분번호'].astype(str)
    df_crawled['가맹점구분번호'] = df_crawled['가맹점구분번호'].astype(str)

    # 완료된 ID 목록 추출
    crawled_ids = set(df_crawled['가맹점구분번호'].unique())
    
    # 원본 데이터 중복 제거
    df_origin_unique = df_origin.drop_duplicates(subset=['가맹점구분번호'])
    
    total_stores = len(df_origin_unique)
    done_stores = len(crawled_ids)

    print(f"   전체 가맹점 수: {total_stores}개")
    print(f"   크롤링 완료 수: {done_stores}개")

    # 미검색 가맹점 필터링
    missing_df = df_origin_unique[~df_origin_unique['가맹점구분번호'].isin(crawled_ids)].copy()
    
    missing_count = len(missing_df)
    
    if missing_count == 0:
        print("\n모든 가맹점 크롤링이 완료되었습니다.")
        return

    print(f"   아직 검색되지 않은 가맹점: {missing_count}개")

    # 미검색 가맹점 목록 최종 저장
    result_df = missing_df[['가맹점구분번호', '가맹점명', '가맹점주소']]
    result_df.to_excel(output_path, index=False)

if __name__ == "__main__":
    main()