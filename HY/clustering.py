"""
K-Prototypes 클러스터링 모듈
가맹점 특성에 따른 군집 분석
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def prepare_clustering_data(df):
    """
    클러스터링을 위한 데이터 준비
    가맹점별로 집계하여 1행으로 만듦
    """
    # 특수값 처리 (-999999.9 → NaN)
    df_clean = df.copy()
    special_value_cols = [
        '취소율 구간', '배달매출금액 비율', '동일 상권 내 해지 가맹점 비중',
        '남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중',
        '남성 50대 고객 비중', '남성 60대이상 고객 비중',
        '여성 20대이하 고객 비중', '여성 30대 고객 비중', '여성 40대 고객 비중',
        '여성 50대 고객 비중', '여성 60대이상 고객 비중',
        '재방문 고객 비중', '신규 고객 비중',
        '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율'
    ]
    for col in special_value_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(-999999.9, np.nan)

    # 가맹점별 집계
    agg_funcs = {
        # 기본 정보 (최빈값/첫번째 값)
        '가맹점명': 'first',
        '브랜드구분코드': 'first',
        '업종': 'first',
        '상권': 'first',

        # 수치형 - 평균
        '매출금액 구간': 'mean',
        '매출건수 구간': 'mean',
        '객단가 구간': 'mean',
        '유니크 고객 수 구간': 'mean',
        '동일 업종 내 매출 순위 비율': 'mean',
        '동일 상권 내 매출 순위 비율': 'mean',
        '재방문 고객 비중': 'mean',
        '신규 고객 비중': 'mean',

        # 고객층
        '남성 20대이하 고객 비중': 'mean',
        '남성 30대 고객 비중': 'mean',
        '남성 40대 고객 비중': 'mean',
        '남성 50대 고객 비중': 'mean',
        '남성 60대이상 고객 비중': 'mean',
        '여성 20대이하 고객 비중': 'mean',
        '여성 30대 고객 비중': 'mean',
        '여성 40대 고객 비중': 'mean',
        '여성 50대 고객 비중': 'mean',
        '여성 60대이상 고객 비중': 'mean',

        # 고객 유형
        '거주 이용 고객 비율': 'mean',
        '직장 이용 고객 비율': 'mean',
        '유동인구 이용 고객 비율': 'mean',

        # 기간 정보
        '기준년월': 'count'
    }

    store_df = df_clean.groupby('가맹점구분번호').agg(agg_funcs).reset_index()
    store_df.rename(columns={'기준년월': '데이터개월수'}, inplace=True)

    # 매출 변동성 (표준편차) 추가
    sales_std = df.groupby('가맹점구분번호')['매출금액 구간'].std().reset_index()
    sales_std.columns = ['가맹점구분번호', '매출변동성']
    store_df = store_df.merge(sales_std, on='가맹점구분번호')

    return store_df


def run_kprototypes_clustering(store_df, n_clusters=5):
    """
    K-Prototypes 클러스터링 실행
    범주형 + 수치형 혼합 데이터 처리
    """
    try:
        from kmodes.kprototypes import KPrototypes
    except ImportError:
        return None, "kmodes 라이브러리가 설치되어 있지 않습니다. pip install kmodes 실행 필요"

    # 클러스터링에 사용할 변수 선택
    numeric_cols = [
        '매출금액 구간', '매출건수 구간', '객단가 구간',
        '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율',
        '재방문 고객 비중', '매출변동성'
    ]

    categorical_cols = ['업종', '상권']

    # 결측치 처리
    df_cluster = store_df.copy()

    for col in numeric_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    for col in categorical_cols:
        df_cluster[col] = df_cluster[col].fillna('기타')

    # 수치형 변수 정규화
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(df_cluster[numeric_cols])

    # 범주형 변수 인코딩 (문자열 그대로 사용)
    categorical_data = df_cluster[categorical_cols].values

    # 데이터 결합
    X = np.hstack([numeric_data, categorical_data])

    # 범주형 컬럼 인덱스 (뒤쪽에 위치)
    categorical_indices = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))

    # K-Prototypes 실행
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=0, random_state=42)
    clusters = kproto.fit_predict(X, categorical=categorical_indices)

    df_cluster['클러스터'] = clusters

    return df_cluster, kproto


def run_kmeans_clustering(store_df, n_clusters=5):
    """
    K-Means 클러스터링 (수치형 변수만 사용)
    kmodes가 없을 때 대안
    """
    from sklearn.cluster import KMeans

    numeric_cols = [
        '매출금액 구간', '매출건수 구간', '객단가 구간',
        '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율',
        '재방문 고객 비중', '매출변동성'
    ]

    df_cluster = store_df.copy()

    for col in numeric_cols:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    # 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cluster[numeric_cols])

    # K-Means 실행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df_cluster['클러스터'] = clusters

    return df_cluster, kmeans


def get_cluster_profiles(df_clustered):
    """
    각 클러스터의 프로필 생성
    """
    profile_cols = [
        '매출금액 구간', '매출건수 구간', '객단가 구간',
        '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율',
        '재방문 고객 비중', '매출변동성'
    ]

    profiles = df_clustered.groupby('클러스터')[profile_cols].mean()

    # 클러스터 특성 해석 (상대적 기준)
    cluster_names = []

    # 전체 평균 계산
    avg_sales = profiles['매출금액 구간'].mean()
    avg_revisit = profiles['재방문 고객 비중'].mean()
    avg_rank = profiles['동일 업종 내 매출 순위 비율'].mean()

    for idx in profiles.index:
        row = profiles.loc[idx]

        high_sales = row['매출금액 구간'] >= avg_sales + 0.5
        low_sales = row['매출금액 구간'] < avg_sales - 0.5
        high_revisit = row['재방문 고객 비중'] >= avg_revisit + 5
        good_rank = row['동일 업종 내 매출 순위 비율'] < avg_rank - 10  # 낮을수록 좋음

        if high_sales and good_rank:
            name = "⭐ 고성과 우수"
        elif high_sales and high_revisit:
            name = "💎 충성고객 기반"
        elif high_sales:
            name = "📊 고매출"
        elif high_revisit and not low_sales:
            name = "💚 재방문 강점"
        elif low_sales and not good_rank:
            name = "⚠️ 성장 필요"
        elif good_rank:
            name = "🏆 업종 내 강자"
        else:
            name = "🔄 일반형"

        cluster_names.append(name)

    profiles['클러스터명'] = cluster_names

    return profiles


def assign_cluster_to_store(store_id, df_clustered):
    """
    특정 가맹점의 클러스터 정보 반환
    """
    store_row = df_clustered[df_clustered['가맹점구분번호'] == store_id]

    if len(store_row) == 0:
        return None

    return store_row.iloc[0]


def get_similar_stores(store_id, df_clustered, n=5):
    """
    같은 클러스터 내 유사 가맹점 반환
    """
    store_row = df_clustered[df_clustered['가맹점구분번호'] == store_id]

    if len(store_row) == 0:
        return pd.DataFrame()

    cluster = store_row['클러스터'].iloc[0]
    same_cluster = df_clustered[
        (df_clustered['클러스터'] == cluster) &
        (df_clustered['가맹점구분번호'] != store_id)
    ]

    # 매출금액 구간 기준으로 유사한 순서
    store_sales = store_row['매출금액 구간'].iloc[0]
    same_cluster['유사도'] = abs(same_cluster['매출금액 구간'] - store_sales)
    similar = same_cluster.nsmallest(n, '유사도')

    return similar[['가맹점명', '브랜드구분코드', '업종', '상권', '매출금액 구간', '재방문 고객 비중']]
