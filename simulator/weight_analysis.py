import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def analyze_team_performance_lr(pitcher_df):
    """
    방법 1-LR: 선형 회귀를 이용해 팀 승률과 가장 관계가 깊은 가중치를 분석합니다.
    """
    print("\n" + "="*70)
    print("방법 1: 팀 승률 기반 가중치 분석 (Linear Regression)")
    print("="*70)

    try:
        team_df = pd.read_excel('졸프용 데이터베이스.xlsx', sheet_name='팀 성적')
    except FileNotFoundError:
        print("'팀 성적' 시트를 찾을 수 없습니다.")
        return None

    # 데이터 준비
    pitcher_team_stats = pitcher_df.groupby(['연도', '팀']).agg({
        '보정FIP': 'mean',
        '보정피안타': 'mean',
        '이닝': 'sum'
    }).reset_index()
    pitcher_team_stats = pitcher_team_stats[pitcher_team_stats['이닝'] >= 1000]
    
    merged_df = pitcher_team_stats.merge(team_df, on=['연도', '팀'], how='inner')
    
    if '승률' not in merged_df.columns and ('승' in merged_df.columns and '패' in merged_df.columns):
        merged_df['승률'] = merged_df['승'] / (merged_df['승'] + merged_df['패'])

    analysis_df = merged_df[['승률', '보정FIP', '보정피안타']].dropna()
    print(f"분석 대상: {len(analysis_df)} 팀-시즌 데이터")

    # 선형 회귀 모델 적용
    X = analysis_df[['보정FIP', '보정피안타']]
    y = analysis_df['승률']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 정규화된 가중치 계산
    coefficients = model.coef_
    abs_coeffs = np.abs(coefficients)
    normalized_weights = abs_coeffs / np.sum(abs_coeffs)
    
    fip_weight, hit_rate_weight = normalized_weights
    
    print(f"분석 결과 (정규화 가중치):")
    print(f"   - 보정FIP: {fip_weight:.2%}")
    print(f"   - 보정피안타: {hit_rate_weight:.2%}")
    
    return fip_weight, hit_rate_weight

def analyze_individual_performance_lr(pitcher_df):
    """
    방법 2-LR (기존 방법 3): 선형 회귀를 이용해 개별 투수 성과(ERA*)를 
    가장 잘 설명하는 가중치를 분석합니다.
    """
    print("\n" + "="*70)
    print("방법 2: 개별 성과 기반 가중치 분석 (Linear Regression)")
    print("="*70)

    # 데이터 준비
    analysis_df = pitcher_df[['ERA*', '보정FIP', '보정피안타', '이닝']].copy()
    analysis_df.dropna(inplace=True)
    analysis_df = analysis_df[analysis_df['이닝'] >= 50]
    print(f"분석 대상: {len(analysis_df)}명 투수 데이터")

    # 선형 회귀 모델 적용
    X = analysis_df[['보정FIP', '보정피안타']]
    y = analysis_df['ERA*']

    model = LinearRegression()
    model.fit(X, y)

    # 정규화된 가중치 계산
    coefficients = model.coef_
    abs_coeffs = np.abs(coefficients)
    normalized_weights = abs_coeffs / np.sum(abs_coeffs)
    
    fip_weight, hit_rate_weight = normalized_weights

    print(f"분석 결과 (정규화 가중치):")
    print(f"   - 보정FIP: {fip_weight:.2%}")
    print(f"   - 보정피안타: {hit_rate_weight:.2%}")
    
    return fip_weight, hit_rate_weight

def calculate_optimal_weights():
    """
    메인 분석 로직을 실행하고 결과를 종합합니다.
    """
    
    print("="*70)
    print(" KBO 투수 지표 최적 가중치 분석 (선형 회귀 기반)")
    print("="*70)
    
    try:
        pitcher_df = pd.read_excel('졸프용 데이터베이스.xlsx', sheet_name='투수보정')
    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        return

    # 각 방법론 실행
    team_weights = analyze_team_performance_lr(pitcher_df)
    individual_weights = analyze_individual_performance_lr(pitcher_df)

    # 최종 결과 종합
    print("\n" + "="*70)
    print("최종 결과 종합")
    print("="*70)

    if team_weights and individual_weights:
        avg_fip_weight = np.mean([team_weights[0], individual_weights[0]])
        avg_hit_rate_weight = np.mean([team_weights[1], individual_weights[1]])
        
        print("각 방법론별 최적 가중치:")
        print(f"  - 팀 승률 기반: FIP {team_weights[0]:.2f}, 피안타 {team_weights[1]:.2f}")
        print(f"  - 개별 성과 기반: FIP {individual_weights[0]:.2f}, 피안타 {individual_weights[1]:.2f}")
        
        print("\n" + "-"*40)
        print("최종 권장 가중치 (두 방법의 평균):")
        print(f"   - 보정FIP: {avg_fip_weight:.2%}")
        print(f"   - 보정피안타: {avg_hit_rate_weight:.2%}")
        print(f"   => 최종 비율: {avg_fip_weight:.2f} : {avg_hit_rate_weight:.2f}")
        print("-" * 40)
        
        return avg_fip_weight, avg_hit_rate_weight
    else:
        print("분석을 완료하지 못했습니다.")
        return None, None

if __name__ == "__main__":
    calculate_optimal_weights()