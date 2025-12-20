import pandas as pd
import os

def main():

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')

        master_path = os.path.join(data_dir, '최종 데이터셋.csv')
        main_result_path = os.path.join(data_dir, '카카오크롤링_최종_통합본.xlsx')
        supp_result_path = os.path.join(data_dir, 'kakao_ratings_supplement.xlsx')

        final_output_path = os.path.join(data_dir, '카카오크롤링_완전통합본.xlsx')
        missing_output_path = os.path.join(data_dir, '최종_미검색_가맹점.xlsx')
        
    except Exception as e:
        print(f"경로 설정 오류: {e}")
        return

    # --- 2. 데이터 불러오기 ---
    print("📂 파일 로딩 중...")
    
    # 2-1. 전체 명부
    try:
        df_master = pd.read_csv(master_path)
    except:
        df_master = pd.read_excel(master_path)
    
    # 2-2. 기존 통합본
    try:
        df_main = pd.read_excel(main_result_path)
    except:
        df_main = pd.read_csv(main_result_path)

    # 2-3. 추가 발굴본
    df_supp = pd.DataFrame()
    if os.path.exists(supp_result_path):
        try:
            df_supp = pd.read_excel(supp_result_path)
        except:
            df_supp = pd.read_csv(supp_result_path)

    # --- 3. 컬럼명 통일 (종합별점 -> 별점) ---
    print("🔧 컬럼명 통일 중 ('종합별점' -> '별점')...")
    
    if '종합별점' in df_main.columns:
        df_main.rename(columns={'종합별점': '별점'}, inplace=True)
        
    if not df_supp.empty and '종합별점' in df_supp.columns:
        df_supp.rename(columns={'종합별점': '별점'}, inplace=True)

    # --- 4. 데이터 병합 ---
    print("🔗 데이터 합치는 중...")

    # 가맹점구분번호 문자열 변환
    df_main['가맹점구분번호'] = df_main['가맹점구분번호'].astype(str)
    if not df_supp.empty:
        df_supp['가맹점구분번호'] = df_supp['가맹점구분번호'].astype(str)
        df_final = pd.concat([df_main, df_supp], ignore_index=True)
    else:
        df_final = df_main.copy()

    # 중복 제거 (최신 데이터 우선)
    df_final = df_final.drop_duplicates(subset=['가맹점구분번호'], keep='last')
    
    # 최종 저장
    df_final.to_excel(final_output_path, index=False)
    print(f"   ✅ 통합 완료! '{final_output_path}' (총 {len(df_final)}개)")

    # --- 5. 미검색 가맹점 추출 ---
    print("🔍 남은 가맹점 확인 중...")
    
    df_master['가맹점구분번호'] = df_master['가맹점구분번호'].astype(str)
    df_master_unique = df_master.drop_duplicates(subset=['가맹점구분번호'])
    
    crawled_ids = set(df_final['가맹점구분번호'])
    
    df_missing = df_master_unique[~df_master_unique['가맹점구분번호'].isin(crawled_ids)].copy()
    
    if len(df_missing) > 0:
        df_missing_save = df_missing[['가맹점구분번호', '가맹점명', '가맹점주소']]
        df_missing_save.to_excel(missing_output_path, index=False)
        print(f"   🚨 미검색 가맹점: {len(df_missing)}개 -> '{missing_output_path}' 저장됨")
    else:
        print("   🎉 모든 가맹점 처리 완료!")

if __name__ == "__main__":
    main()