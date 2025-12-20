# 4차 시도 후 가맹점 통합

import pandas as pd
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')

        main_path = os.path.join(data_dir, '카카오크롤링_완전통합본.xlsx')
        success_path = os.path.join(data_dir, '검색성공_20251201_1350.xlsx')
        fail_path = os.path.join(data_dir, '검색실패_20251201_1350.xlsx')
        
        output_path = os.path.join(data_dir, '카카오크롤링_진짜진짜완전최종이제그만.xlsx')
        
    except Exception as e:
        print(f"경로 오류: {e}")
        return
    
    try:
        df_main = pd.read_excel(main_path)
    except:
        df_main = pd.read_csv(main_path)

    try:
        df_success = pd.read_excel(success_path)
    except:
        df_success = pd.DataFrame()

    try:
        df_fail = pd.read_excel(fail_path)
    except:
        df_fail = pd.DataFrame()

    # 데이터 포맷 통일
    
    # [검색 성공 파일] 가맹점명 -> 식당명 변경
    if not df_success.empty:
        if '가맹점명' in df_success.columns:
            df_success.rename(columns={'가맹점명': '식당명'}, inplace=True)
        # 필요한 컬럼만 선택
        df_success = df_success[['가맹점구분번호', '식당명', '별점', '카테고리평가']].copy()

    # [검색 실패 파일] 가맹점명 -> 식당명 변경, 별점/카테고리에 NaN 채우기
    if not df_fail.empty:
        if '가맹점명' in df_fail.columns:
            df_fail.rename(columns={'가맹점명': '식당명'}, inplace=True)
        
        # 빈 컬럼 추가
        df_fail['별점'] = None
        df_fail['카테고리평가'] = None
        
        # 필요한 컬럼만 선택
        df_fail = df_fail[['가맹점구분번호', '식당명', '별점', '카테고리평가']].copy()
    
    # ID 문자열 통일
    df_main['가맹점구분번호'] = df_main['가맹점구분번호'].astype(str)
    if not df_success.empty: df_success['가맹점구분번호'] = df_success['가맹점구분번호'].astype(str)
    if not df_fail.empty: df_fail['가맹점구분번호'] = df_fail['가맹점구분번호'].astype(str)

    # 합치기
    df_final = pd.concat([df_main, df_success, df_fail], ignore_index=True)
    
    # 중복 제거
    df_final = df_final.drop_duplicates(subset=['가맹점구분번호'], keep='last')

    # 저장
    df_final.to_excel(output_path, index=False)
    
    print(f" 최종 완료")
    print(f"   총 합계: {len(df_final)}개")
    print(f"   저장 위치: {output_path}")

if __name__ == "__main__":
    main()