import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# --- 페이지 설정 ---
st.set_page_config(
    page_title="KBO 투수 평가 대시보드",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 데이터 로딩 ---
# 데이터 파일들은 app.py와 같은 폴더에 있어야 합니다.
@st.cache_data
def load_data():
    try:
        original_file = 'all_pitchers_simulation_results.csv'
        optimized_file = 'pitchers_simulation_optimized_weights.csv'
        
        original_df = pd.read_csv(original_file)
        optimized_df = pd.read_csv(optimized_file)

        # '투수명'과 '연도'를 기준으로 데이터 병합
        merged_df = pd.merge(
            original_df, optimized_df, on=['투수명', '연도'], suffixes=('_orig', '_opt')
        )
        
        # 중복 컬럼 정리
        merged_df.rename(columns={'ERA*_orig': 'ERA*', 'PAI(100)': 'PAI_orig', 'PAI(100)_optimized': 'PAI'}, inplace=True)
        
        # 필요한 컬럼만 선택하고 순서 지정
        cols_to_keep = [
            '연도', '투수명', 'PAI', 'ERA*', 'PAI_orig', 'FIP*_orig',
            'K/9_orig', 'BB/9_orig', 'HR/9_orig', '피OPS_orig'
        ]
        # 일부 컬럼 이름 변경
        merged_df = merged_df[cols_to_keep].rename(columns={
            'FIP*_orig': 'FIP*', 'K/9_orig': 'K/9', 'BB/9_orig': 'BB/9',
            'HR/9_orig': 'HR/9', '피OPS_orig': '피OPS'
        })
        
        # 팀 정보 추가
        pitcher_df = pd.read_excel('졸프용 데이터베이스.xlsx', sheet_name='투수보정')
        merged_df = pd.merge(merged_df, pitcher_df[['선수명', '연도', '팀']], left_on=['투수명', '연도'], right_on=['선수명', '연도'], how='left')

        return merged_df.copy()

    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 'all_pitchers_simulation_results.csv'와 'pitchers_simulation_optimized_weights.csv' 파일이 현재 디렉토리에 있는지 확인하세요.")
        return None

df = load_data()

st.sidebar.title("⚾ KBO 투수 평가 대시보드")
page = st.sidebar.radio("페이지 선택", ["프로젝트 소개", "PAI 랭킹 대시보드", "모델 성능 비교"])

# --- 페이지별 콘텐츠 ---

# 1. 프로젝트 소개 페이지
if page == "프로젝트 소개":
    st.title("투수-타자 맞대결 시뮬레이터 기반 KBO 투수 종합지표(PAI) 개발")
    st.markdown("---")
    
    st.header("1. 프로젝트 배경 및 목표")
    st.markdown("""
    야구에서 타자의 능력을 종합적으로 평가하는 wRC+와 같은 지표는 널리 사용되지만, 투수의 능력을 객관적으로 평가하는 단일 종합 지표는 부족한 실정입니다.
    
    본 프로젝트는 KBO(한국프로야구)의 방대한 데이터를 기반으로 투수와 타자의 맞대결 시뮬레이터를 개발하고, 그 결과를 바탕으로 투수의 가치를 종합적으로 평가할 수 있는 새로운 지표인 PAI(Pitcher Assessment Index)를 개발하는 것을 목표로 합니다.
    """)
    
    st.header("2. PAI(Pitcher Assessment Index)란?")
    st.markdown("""
    PAI는 투수의 핵심적인 능력을 나타내는 두 가지 세부 지표를 결합하여 만든 종합 평가 지표입니다.
    
    - 보정FIP (Adjusted FIP): 홈런, 볼넷, 삼진 제어 능력을 평가하는 FIP(수비 무관 평균자책점)를 리그 평균 대비 100을 기준으로 보정한 값입니다.
    - 보정피안타 (Adjusted Hits Allowed): 투수의 안타 억제 능력을 리그 평균 대비 100을 기준으로 보정한 값입니다.
    
    초기 모델은 보정FIP에 70%, 보정피안타에 30%의 가중치를 부여하여 PAI를 산출했습니다.
    
    최적 모델을 찾기위해 선형 회귀 분석을 통해 실제 팀 승률과 개인 성과(ERA*)를 가장 잘 설명하는 두 지표의 최적 가중치를 계산하여 PAI를 산출해서, 기존 모델과 시뮬레이션 결과를 비교했습니다.
    """)

    st.info("""
    PAI 해석:
    - PAI는 100을 기준으로 하며, 높을수록 좋은 투수임을 의미합니다.
    - 예: PAI가 120인 투수는 리그 평균 투수보다 20% 더 뛰어난 성과를 냈다고 해석할 수 있습니다.
    """)

# 2. PAI 랭킹 대시보드 페이지
elif page == "PAI 랭킹 대시보드":
    st.title("PAI 랭킹 대시보드")
    st.markdown("연도, 팀, 선수별 PAI 랭킹 및 주요 스탯")

    if df is not None:
        # --- 필터링 옵션 ---
        st.sidebar.markdown("---")
        st.sidebar.header("랭킹 필터")
        
        # 연도 필터
        sorted_years = sorted(df['연도'].unique(), reverse=True)
        selected_year = st.sidebar.selectbox("연도 선택", sorted_years)

        # 선수 검색
        player_name = st.sidebar.text_input("선수명 검색 (일부만 입력해도 가능)")
        
        # 데이터 필터링
        filtered_df = df[df['연도'] == selected_year]
        if player_name:
            filtered_df = filtered_df[filtered_df['투수명'].str.contains(player_name, case=False, na=False)]

        st.subheader(f"**{selected_year}년 시즌 투수 PAI 랭킹**")
        
        # 데이터프레임 표시 (인덱스 제거 및 소수점 정리)
        st.dataframe(filtered_df.sort_values(by="PAI", ascending=False).reset_index(drop=True).style.format({
            'PAI': '{:.2f}',
            'ERA*': '{:.2f}',
            'PAI_orig': '{:.2f}',
            'FIP*': '{:.2f}',
            'K/9': '{:.2f}',
            'BB/9': '{:.2f}',
            'HR/9': '{:.2f}',
            '피OPS': '{:.3f}'
        }))

        # 다운로드 버튼
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="필터링된 데이터 다운로드 (CSV)",
            data=csv,
            file_name=f'{selected_year}_pitcher_rankings.csv',
            mime='text/csv',
        )

# 3. 모델 성능 비교 페이지
elif page == "모델 성능 비교":
    st.title("모델 성능 비교 분석")
    st.markdown("초기에 임의로 설정한 가중치(보정FIP 70%, 보정피안타 30%)와 최적화된 가중치 기반 모델의 성능을 비교합니다. 선형 회귀 분석을 통해 얻은 최적 가중치는 보정FIP 34%, 보정피안타 66%로 나타났습니다.")
    st.markdown("---")

    if df is not None:
        # ERA*가 0 이하인 비현실적 데이터 제외
        analysis_data = df[df['ERA*'] > 0].copy()

        # 상관계수 계산
        corr_orig = analysis_data['PAI_orig'].corr(analysis_data['ERA*'])
        corr_opt = analysis_data['PAI'].corr(analysis_data['ERA*'])
        improvement = corr_opt - corr_orig

        st.header("1. 실제 성과(ERA*)와의 예측력 비교")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="기존 모델 PAI vs ERA* 상관계수", value=f"{corr_orig:.4f}")
        with col2:
            st.metric(label="신규 모델 PAI vs ERA* 상관계수", value=f"{corr_opt:.4f}", delta=f"{improvement:.4f} 개선")

        st.success(f"**결론:** 신규 모델이 기존 모델보다 실제 투수 성적(ERA\*)을 **{improvement/corr_orig:.2%}** 더 정확하게 예측합니다.")
        st.markdown("_(상관계수의 절대값이 클수록 예측력이 높음을 의미)_")
        st.markdown("---")

        st.header("2. 분석 결과 시각화")
        try:
            # 이미지 파일 로드
            dist_img = Image.open('pai_distribution_comparison.png')
            corr_img = Image.open('pai_correlation_comparison.png')
            
            st.subheader("PAI 점수 분포 비교")
            st.image(dist_img, caption="두 모델의 PAI 점수 분포 비교")
            st.markdown("신규 모델의 PAI 분포가 더 안정적인 형태를 보이며, 이는 투수들을 더 일관된 척도로 평가함을 시사합니다.")

            st.subheader("모델별 예측력 비교 시각화")
            st.image(corr_img, caption="모델별 PAI와 실제 성과(ERA*)의 관계")
            st.markdown("신규 모델의 회귀선(빨간선)이 데이터에 더 잘 적합되는 경향을 보여, ERA* 예측력이 더 높음을 시각적으로 확인할 수 있습니다.")

        except FileNotFoundError:
            st.warning("시각화 이미지 파일을 찾을 수 없습니다. 'pai_distribution_comparison.png'와 'pai_correlation_comparison.png' 파일이 현재 디렉토리에 있는지 확인하세요.")