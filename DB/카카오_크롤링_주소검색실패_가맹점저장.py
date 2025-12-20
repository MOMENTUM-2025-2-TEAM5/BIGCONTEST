# 1차 주소 검색 시도 후, 실패한 가맹점들 별도로 저장하는 코드

import pandas as pd
import os

def check_failures():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, 'data')
        file_path = os.path.join(data_dir, 'kakao_store_list.csv')
        
        print(f"파일 읽는 중: {file_path}")
        df = pd.read_csv(file_path)
        
    except FileNotFoundError:
        print("❌ 오류: 'kakao_store_list.csv' 파일이 아직 생성되지 않았거나 경로가 틀렸습니다.")
        return

    # 개수 확인 
    fail_df = df[df['카카오맵식당명'] == '못찾음']
    
    # '못찾음'이 아닌 것 (성공)
    success_df = df[df['카카오맵식당명'] != '못찾음']

    total_count = len(df)
    fail_count = len(fail_df)
    success_count = len(success_df)

    # 결과 출력
    print(f"총 시도 횟수 : {total_count}개")
    print(f"성공 (찾음) : {success_count}개")
    print(f"실패 (못찾음): {fail_count}개")
    
    # 실패한 목록만 따로 엑셀로 저장
    if fail_count > 0:
        save_path = os.path.join(data_dir, 'search_failed_list.xlsx')
        fail_df.to_excel(save_path, index=False)

if __name__ == "__main__":
    check_failures()