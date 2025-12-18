"""
가맹점 분석 대시보드 + AI 챗봇 통합 버전
- 기존 dashboard.py의 분석 기능
- chatbot_module.py의 AI 컨설팅 챗봇
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 클러스터링 모듈 임포트
from clustering import (
    prepare_clustering_data,
    run_kprototypes_clustering,
    run_kmeans_clustering,
    get_cluster_profiles,
    get_similar_stores,
)

# 페이지 설정
st.set_page_config(
    page_title="가맹점 분석 대시보드 + AI 컨설턴트", page_icon="🤖", layout="wide"
)

# 커스텀 CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== 데이터 로드 함수 ====================
@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    df = pd.read_excel("public/최종 데이터셋_v2.xlsx")

    # 특수값 처리 (-999999.9 → NaN)
    special_value_cols = [
        "취소율 구간",
        "배달매출금액 비율",
        "동일 상권 내 해지 가맹점 비중",
        "남성 20대이하 고객 비중",
        "남성 30대 고객 비중",
        "남성 40대 고객 비중",
        "남성 50대 고객 비중",
        "남성 60대이상 고객 비중",
        "여성 20대이하 고객 비중",
        "여성 30대 고객 비중",
        "여성 40대 고객 비중",
        "여성 50대 고객 비중",
        "여성 60대이상 고객 비중",
        "재방문 고객 비중",
        "신규 고객 비중",
        "거주 이용 고객 비율",
        "직장 이용 고객 비율",
        "유동인구 이용 고객 비율",
    ]

    for col in special_value_cols:
        if col in df.columns:
            df[col] = df[col].replace(-999999.9, np.nan)

    # 폐업 여부
    df["폐업여부"] = (df["폐업일"].notna() & (df["폐업일"] != -999999.9)).astype(int)

    # 별점 숫자로 변환
    df["별점_숫자"] = pd.to_numeric(df["별점"], errors="coerce")

    # 기준년월 문자열 변환
    df["기준년월_str"] = (
        df["기준년월"].astype(str).str[:4] + "-" + df["기준년월"].astype(str).str[4:]
    )

    return df


@st.cache_data
def get_store_summary(df):
    """가맹점별 요약 데이터 생성"""
    summary = (
        df.groupby("가맹점구분번호")
        .agg(
            {
                "가맹점명": "first",
                "브랜드구분코드": "first",
                "브랜드이름": "first",
                "업종": "first",
                "상권": "first",
                "가맹점지역": "first",
                "가맹점주소": "first",
                "별점_숫자": "mean",
                "매출금액 구간": "mean",
                "매출건수 구간": "mean",
                "객단가 구간": "mean",
                "재방문 고객 비중": "mean",
                "신규 고객 비중": "mean",
                "동일 업종 내 매출 순위 비율": "mean",
                "동일 상권 내 매출 순위 비율": "mean",
                "기준년월": ["min", "max", "count"],
                "폐업여부": "max",
            }
        )
        .reset_index()
    )

    # 컬럼명 정리
    summary.columns = [
        "가맹점구분번호",
        "가맹점명",
        "브랜드구분코드",
        "브랜드이름",
        "업종",
        "상권",
        "가맹점지역",
        "가맹점주소",
        "평균별점",
        "평균매출구간",
        "평균매출건수구간",
        "평균객단가구간",
        "평균재방문비중",
        "평균신규고객비중",
        "평균업종내순위",
        "평균상권내순위",
        "첫월",
        "마지막월",
        "데이터개월수",
        "폐업여부",
    ]

    return summary


# ==================== 챗봇 초기화 ====================
@st.cache_resource
def get_chatbot():
    """챗봇 매니저 초기화 (싱글톤)"""
    try:
        from chatbot_module import ChatbotManager

        chatbot = ChatbotManager()
        chatbot.initialize()
        return chatbot, None
    except ImportError as e:
        return None, f"챗봇 모듈 로드 실패: {e}"
    except ValueError as e:
        return None, f"API 키 오류: {e}"
    except Exception as e:
        return None, f"챗봇 초기화 실패: {e}"


# ==================== 대시보드 페이지 ====================
def dashboard_page(df, store_summary):
    """대시보드 메인 페이지"""
    st.title("📊 가맹점 분석 대시보드")
    st.markdown("서울 성동구 가맹점 데이터 분석 (2023.01 ~ 2024.12)")

    # ==================== 사이드바 필터 ====================
    st.sidebar.header("🔍 필터 설정")

    # 업종 필터
    업종_list = ["전체"] + sorted(df["업종"].unique().tolist())
    selected_업종 = st.sidebar.selectbox("업종 선택", 업종_list)

    # 상권 필터
    상권_list = ["전체"] + sorted(df["상권"].unique().tolist())
    selected_상권 = st.sidebar.selectbox("상권 선택", 상권_list)

    # 브랜드 필터 (검색 가능)
    브랜드_list = ["전체"] + sorted(df["브랜드구분코드"].unique().tolist())
    selected_브랜드 = st.sidebar.selectbox(
        "브랜드 선택 (검색 가능)",
        브랜드_list,
        help="프랜차이즈 브랜드를 선택하면 해당 브랜드의 모든 가맹점을 볼 수 있습니다",
    )

    # 별점 범위
    별점_range = st.sidebar.slider(
        "별점 범위", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.1
    )

    # 필터 적용
    filtered_summary = store_summary.copy()

    if selected_업종 != "전체":
        filtered_summary = filtered_summary[filtered_summary["업종"] == selected_업종]

    if selected_상권 != "전체":
        filtered_summary = filtered_summary[filtered_summary["상권"] == selected_상권]

    if selected_브랜드 != "전체":
        filtered_summary = filtered_summary[
            filtered_summary["브랜드구분코드"] == selected_브랜드
        ]

    filtered_summary = filtered_summary[
        (filtered_summary["평균별점"].fillna(0) >= 별점_range[0])
        & (filtered_summary["평균별점"].fillna(5) <= 별점_range[1])
    ]

    # ==================== 메인 화면 ====================

    # 필터 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("필터된 가맹점 수", f"{len(filtered_summary):,}개")
    with col2:
        avg_rating = filtered_summary["평균별점"].mean()
        st.metric(
            "평균 별점", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "N/A"
        )
    with col3:
        avg_sales = filtered_summary["평균매출구간"].mean()
        st.metric(
            "평균 매출구간", f"{avg_sales:.1f}" if not pd.isna(avg_sales) else "N/A"
        )
    with col4:
        avg_revisit = filtered_summary["평균재방문비중"].mean()
        st.metric(
            "평균 재방문율",
            f"{avg_revisit:.1f}%" if not pd.isna(avg_revisit) else "N/A",
        )

    st.markdown("---")

    # ==================== 가맹점 선택 ====================
    st.subheader("🏪 가맹점 선택")

    if len(filtered_summary) == 0:
        st.warning("선택한 필터 조건에 맞는 가맹점이 없습니다.")
        return

    # 가맹점 선택 드롭다운
    store_options = filtered_summary.apply(
        lambda x: f"{x['가맹점명']} ({x['브랜드구분코드']}) - {x['상권']}", axis=1
    ).tolist()

    selected_store_idx = st.selectbox(
        "분석할 가맹점을 선택하세요",
        range(len(store_options)),
        format_func=lambda x: store_options[x],
    )

    selected_store = filtered_summary.iloc[selected_store_idx]
    store_id = selected_store["가맹점구분번호"]

    # 세션에 선택된 가맹점 저장 (챗봇에서 활용)
    st.session_state["selected_store"] = selected_store.to_dict()

    # 해당 가맹점의 시계열 데이터
    store_data = df[df["가맹점구분번호"] == store_id].sort_values("기준년월")

    # ==================== 가맹점 상세 정보 ====================
    st.markdown("---")
    st.subheader("📋 가맹점 상세 정보")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 기본 정보")
        st.write(f"**가맹점구분번호:** {selected_store['가맹점구분번호']}")
        st.write(f"**가맹점명:** {selected_store['가맹점명']}")
        st.write(f"**브랜드:** {selected_store['브랜드구분코드']}")
        if pd.notna(selected_store["브랜드이름"]):
            st.write(f"**브랜드 이름:** {selected_store['브랜드이름']}")
        st.write(f"**업종:** {selected_store['업종']}")
        st.write(f"**상권:** {selected_store['상권']}")
        st.write(f"**주소:** {selected_store['가맹점주소']}")
        st.write(
            f"**별점:** {selected_store['평균별점']:.1f}"
            if pd.notna(selected_store["평균별점"])
            else "**별점:** N/A"
        )
        st.write(
            f"**데이터 기간:** {selected_store['첫월']} ~ {selected_store['마지막월']}"
        )
        st.write(f"**데이터 개월수:** {selected_store['데이터개월수']}개월")

    with col2:
        st.markdown("#### 성과 지표 (기간 평균)")

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("평균 매출구간", f"{selected_store['평균매출구간']:.1f}")
            st.metric("평균 객단가구간", f"{selected_store['평균객단가구간']:.1f}")

        with metrics_col2:
            st.metric("업종 내 순위", f"상위 {selected_store['평균업종내순위']:.1f}%")
            st.metric("상권 내 순위", f"상위 {selected_store['평균상권내순위']:.1f}%")

        with metrics_col3:
            revisit = selected_store["평균재방문비중"]
            st.metric(
                "재방문 고객 비중", f"{revisit:.1f}%" if pd.notna(revisit) else "N/A"
            )
            new_cust = selected_store["평균신규고객비중"]
            st.metric(
                "신규 고객 비중", f"{new_cust:.1f}%" if pd.notna(new_cust) else "N/A"
            )

    # ==================== 시계열 차트 ====================
    st.markdown("---")
    st.subheader("📈 월별 추이 분석")

    tab1, tab2, tab3 = st.tabs(["매출 추이", "고객 분석", "순위 변화"])

    with tab1:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("매출금액 구간", "매출건수 구간")
        )

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["매출금액 구간"],
                mode="lines+markers",
                name="매출금액 구간",
                line=dict(color="#1f77b4", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["매출건수 구간"],
                mode="lines+markers",
                name="매출건수 구간",
                line=dict(color="#ff7f0e", width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["재방문 고객 비중"],
                mode="lines+markers",
                name="재방문 고객 비중",
                line=dict(color="#2ca02c", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["신규 고객 비중"],
                mode="lines+markers",
                name="신규 고객 비중",
                line=dict(color="#d62728", width=2),
            )
        )

        fig.update_layout(
            title="고객 구성 변화",
            xaxis_title="기준년월",
            yaxis_title="비중 (%)",
            height=400,
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["동일 업종 내 매출 순위 비율"],
                mode="lines+markers",
                name="업종 내 순위 (%)",
                line=dict(color="#9467bd", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=store_data["기준년월_str"],
                y=store_data["동일 상권 내 매출 순위 비율"],
                mode="lines+markers",
                name="상권 내 순위 (%)",
                line=dict(color="#8c564b", width=2),
            )
        )

        fig.update_layout(
            title="순위 변화 (낮을수록 상위)",
            xaxis_title="기준년월",
            yaxis_title="순위 비율 (%)",
            height=400,
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # ==================== 동일 업종/상권 비교 ====================
    st.markdown("---")
    st.subheader("🔄 비교 분석")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 동일 업종 내 비교")
        same_업종 = store_summary[store_summary["업종"] == selected_store["업종"]]

        fig = go.Figure()
        fig.add_trace(
            go.Box(y=same_업종["평균매출구간"], name="업종 전체", boxpoints="outliers")
        )
        fig.add_trace(
            go.Scatter(
                x=["업종 전체"],
                y=[selected_store["평균매출구간"]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="현재 가맹점",
            )
        )
        fig.update_layout(
            title=f"{selected_store['업종']} 업종 내 매출구간 분포",
            yaxis_title="평균 매출구간",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"**{selected_store['업종']}** 업종 내 총 **{len(same_업종)}개** 가맹점"
        )
        percentile = (
            same_업종["평균매출구간"] <= selected_store["평균매출구간"]
        ).mean() * 100
        st.write(f"현재 가맹점은 상위 **{100-percentile:.1f}%** 위치")

    with col2:
        st.markdown("#### 동일 상권 내 비교")
        same_상권 = store_summary[store_summary["상권"] == selected_store["상권"]]

        fig = go.Figure()
        fig.add_trace(
            go.Box(y=same_상권["평균매출구간"], name="상권 전체", boxpoints="outliers")
        )
        fig.add_trace(
            go.Scatter(
                x=["상권 전체"],
                y=[selected_store["평균매출구간"]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="star"),
                name="현재 가맹점",
            )
        )
        fig.update_layout(
            title=f"{selected_store['상권']} 상권 내 매출구간 분포",
            yaxis_title="평균 매출구간",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"**{selected_store['상권']}** 상권 내 총 **{len(same_상권)}개** 가맹점"
        )
        percentile = (
            same_상권["평균매출구간"] <= selected_store["평균매출구간"]
        ).mean() * 100
        st.write(f"현재 가맹점은 상위 **{100-percentile:.1f}%** 위치")

    # ==================== 프랜차이즈 비교 ====================
    same_brand = store_summary[
        store_summary["브랜드구분코드"] == selected_store["브랜드구분코드"]
    ]

    if len(same_brand) > 1:
        st.markdown("---")
        st.subheader("🏢 프랜차이즈 내 비교")
        st.write(
            f"**{selected_store['브랜드구분코드']}** 브랜드의 가맹점 **{len(same_brand)}개** 비교"
        )

        brand_comparison = same_brand[
            ["가맹점명", "상권", "평균매출구간", "평균별점", "평균재방문비중"]
        ].copy()
        brand_comparison = brand_comparison.sort_values("평균매출구간", ascending=False)
        brand_comparison["순위"] = range(1, len(brand_comparison) + 1)
        brand_comparison = brand_comparison[
            ["순위", "가맹점명", "상권", "평균매출구간", "평균별점", "평균재방문비중"]
        ]

        current_rank = brand_comparison[
            brand_comparison["가맹점명"] == selected_store["가맹점명"]
        ]["순위"].values[0]
        st.write(f"현재 가맹점 순위: **{current_rank}위** / {len(same_brand)}개")

        st.dataframe(
            brand_comparison.style.apply(
                lambda x: [
                    (
                        "background-color: #fffacd"
                        if x["가맹점명"] == selected_store["가맹점명"]
                        else ""
                    )
                    for _ in x
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ==================== 클러스터링 분석 ====================
    st.markdown("---")
    st.subheader("🎯 클러스터 분석")

    @st.cache_data
    def run_clustering(df):
        store_cluster_df = prepare_clustering_data(df)
        try:
            result, model = run_kprototypes_clustering(store_cluster_df, n_clusters=5)
            if result is None:
                result, model = run_kmeans_clustering(store_cluster_df, n_clusters=5)
                method = "K-Means"
            else:
                method = "K-Prototypes"
        except Exception:
            result, model = run_kmeans_clustering(store_cluster_df, n_clusters=5)
            method = "K-Means"
        return result, method

    df_clustered, cluster_method = run_clustering(df)
    cluster_profiles = get_cluster_profiles(df_clustered)

    store_cluster_info = df_clustered[df_clustered["가맹점구분번호"] == store_id]

    if len(store_cluster_info) > 0:
        current_cluster = store_cluster_info["클러스터"].iloc[0]
        cluster_name = cluster_profiles.loc[current_cluster, "클러스터명"]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 이 가맹점의 클러스터")
            st.info(f"**{cluster_name}** (클러스터 {current_cluster})")
            st.caption(f"분석 방법: {cluster_method}")

            same_cluster_count = (df_clustered["클러스터"] == current_cluster).sum()
            st.write(f"같은 클러스터 가맹점: **{same_cluster_count}개**")

        with col2:
            st.markdown("#### 클러스터별 특성")

            profile_display = cluster_profiles[
                [
                    "매출금액 구간",
                    "재방문 고객 비중",
                    "동일 업종 내 매출 순위 비율",
                    "클러스터명",
                ]
            ].copy()
            profile_display.columns = [
                "평균매출구간",
                "재방문비중",
                "업종내순위",
                "클러스터명",
            ]
            profile_display = profile_display.reset_index()

            fig = px.scatter(
                profile_display,
                x="평균매출구간",
                y="재방문비중",
                size="업종내순위",
                color="클러스터명",
                hover_data=["클러스터"],
                title="클러스터 분포 (크기: 업종 내 순위, 작을수록 상위)",
            )

            current_sales = store_cluster_info["매출금액 구간"].iloc[0]
            current_revisit = store_cluster_info["재방문 고객 비중"].iloc[0]
            if pd.notna(current_revisit):
                fig.add_trace(
                    go.Scatter(
                        x=[current_sales],
                        y=[current_revisit],
                        mode="markers",
                        marker=dict(
                            size=20,
                            color="red",
                            symbol="star",
                            line=dict(width=2, color="black"),
                        ),
                        name="현재 가맹점",
                    )
                )

            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 유사 가맹점 (같은 클러스터)")
        similar_stores = get_similar_stores(store_id, df_clustered, n=5)

        if len(similar_stores) > 0:
            st.dataframe(similar_stores, use_container_width=True, hide_index=True)
        else:
            st.info("같은 클러스터 내 다른 가맹점이 없습니다.")

    # ==================== 고객층 분석 ====================
    st.markdown("---")
    st.subheader("👥 고객층 분석")

    latest_data = store_data.iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 성별/연령대 분포")

        age_gender_data = {
            "구분": [
                "남성 20대이하",
                "남성 30대",
                "남성 40대",
                "남성 50대",
                "남성 60대이상",
                "여성 20대이하",
                "여성 30대",
                "여성 40대",
                "여성 50대",
                "여성 60대이상",
            ],
            "비중": [
                latest_data.get("남성 20대이하 고객 비중", 0),
                latest_data.get("남성 30대 고객 비중", 0),
                latest_data.get("남성 40대 고객 비중", 0),
                latest_data.get("남성 50대 고객 비중", 0),
                latest_data.get("남성 60대이상 고객 비중", 0),
                latest_data.get("여성 20대이하 고객 비중", 0),
                latest_data.get("여성 30대 고객 비중", 0),
                latest_data.get("여성 40대 고객 비중", 0),
                latest_data.get("여성 50대 고객 비중", 0),
                latest_data.get("여성 60대이상 고객 비중", 0),
            ],
        }

        age_df = pd.DataFrame(age_gender_data)
        age_df["비중"] = age_df["비중"].fillna(0)

        if age_df["비중"].sum() > 0:
            fig = px.bar(
                age_df,
                x="구분",
                y="비중",
                color="구분",
                title="고객 연령/성별 분포 (최근 월)",
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("고객 연령/성별 데이터가 없습니다.")

    with col2:
        st.markdown("#### 고객 유형 분포")

        customer_type_data = {
            "유형": ["거주 이용", "직장 이용", "유동인구"],
            "비율": [
                latest_data.get("거주 이용 고객 비율", 0),
                latest_data.get("직장 이용 고객 비율", 0),
                latest_data.get("유동인구 이용 고객 비율", 0),
            ],
        }

        type_df = pd.DataFrame(customer_type_data)
        type_df["비율"] = type_df["비율"].fillna(0)

        if type_df["비율"].sum() > 0:
            fig = px.pie(
                type_df, values="비율", names="유형", title="고객 유형 분포 (최근 월)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("고객 유형 데이터가 없습니다.")


# ==================== 챗봇 페이지 ====================
def chatbot_page():
    """AI 컨설턴트 챗봇 페이지"""
    st.title("🤖 AI 컨설턴트")
    st.markdown("가맹점 데이터 분석 및 마케팅 전략을 AI에게 질문하세요!")

    # 챗봇 초기화
    chatbot, error = get_chatbot()

    if error:
        st.error(f"챗봇을 사용할 수 없습니다: {error}")
        st.info(
            "챗봇을 사용하려면 다음을 확인하세요:\n1. .env 파일에 API_KEY 설정\n2. 필요한 패키지 설치 (langchain, langgraph, langchain-google-genai 등)"
        )
        return

    # 선택된 가맹점 정보 표시
    if "selected_store" in st.session_state:
        store = st.session_state["selected_store"]
        with st.expander("📍 현재 선택된 가맹점", expanded=False):
            st.write(
                f"**{store.get('가맹점명', 'N/A')}** ({store.get('브랜드구분코드', 'N/A')})"
            )
            st.write(
                f"업종: {store.get('업종', 'N/A')} | 상권: {store.get('상권', 'N/A')}"
            )
            st.write(
                f"평균 매출구간: {store.get('평균매출구간', 'N/A'):.1f}"
                if store.get("평균매출구간")
                else ""
            )

    # 채팅 기록 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 예시 질문 버튼
    st.markdown("#### 💡 예시 질문")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 매출 분석", use_container_width=True):
            st.session_state.pending_question = (
                "카페 업종의 평균 매출금액 구간은 어떻게 되나요?"
            )

    with col2:
        if st.button("📈 마케팅 전략", use_container_width=True):
            st.session_state.pending_question = (
                "재방문 고객 비중을 높이기 위한 마케팅 전략을 제안해주세요."
            )

    with col3:
        if st.button("🔍 경쟁 분석", use_container_width=True):
            st.session_state.pending_question = (
                "성수 상권의 한식 업종 경쟁 현황을 분석해주세요."
            )

    st.markdown("---")

    # 채팅 기록 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요...")

    # 예시 질문이 있으면 처리
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    if user_input:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    response = chatbot.chat(user_input)
                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        st.session_state.chat_history = []
        st.rerun()


# ==================== 메인 함수 ====================
def main():
    # 데이터 로드
    with st.spinner("데이터를 불러오는 중..."):
        df = load_data()
        store_summary = get_store_summary(df)

    # 사이드바 네비게이션
    st.sidebar.title("🗂️ 메뉴")

    page = st.sidebar.radio(
        "페이지 선택", ["📊 대시보드", "🤖 AI 컨설턴트"], label_visibility="collapsed"
    )

    if page == "📊 대시보드":
        dashboard_page(df, store_summary)
    else:
        chatbot_page()


if __name__ == "__main__":
    main()
