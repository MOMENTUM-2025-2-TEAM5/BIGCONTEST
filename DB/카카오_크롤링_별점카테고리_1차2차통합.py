# 1차,2차로 얻어진 데이터들 통합하여 엑셀로 저장 (카카오크롤링_최종_통합본.xlsx)

import pandas as pd
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')

        file_1_path = os.path.join(data_dir, '카카오크롤링_별점카테고리_1차.xlsx') 
        file_2_path = os.path.join(data_dir, 'kakao_ratings_final.xlsx')   
        output_path = os.path.join(data_dir, '카카오크롤링_최종_통합본.xlsx')  
        
    except Exception as e:
        print(f"경로 설정 오류: {e}")
        return
    
    try:
        df1 = pd.read_excel(file_1_path)
    except:
        df1 = pd.read_csv(file_1_path)
    
    try:
        df2 = pd.read_excel(file_2_path)
    except:
        df2 = pd.read_csv(file_2_path)

    print(f"   1차 파일: {len(df1)}개")
    print(f"   추가 파일: {len(df2)}개")

    # 컬럼명 통일
    if '종합별점' in df2.columns:
        df2.rename(columns={'종합별점': '별점'}, inplace=True)
        print("   ✅ 컬럼명 변경 완료: '종합별점' -> '별점'")

    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # 혹시 모를 ID 중복 제거
    merged_df = merged_df.drop_duplicates(subset=['가맹점구분번호'], keep='last')

    # 최종 저장
    merged_df.to_excel(output_path, index=False)
    print(f"   총 데이터 개수: {len(merged_df)}개")

if __name__ == "__main__":
    main()