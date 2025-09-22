import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def correct_weight_analysis():
    """올바른 접근법: 투수 능력 지표 최적화"""
    print("투수 능력 지표 최적화 분석")
    print("="*60)
    
    # 데이터 로드
    file_path = '졸프용 데이터베이스.xlsx'
    df = pd.read_excel(file_path, sheet_name='투수보정')
    
    print(f"전체 데이터: {df.shape}")
    
    # 방법 1: 팀 성과와의 관계 분석
    print("\n방법 1: 팀 승률과의 관계 분석")
    analyze_team_performance(df)
    
    # 방법 2: 미래 성과 예측력 분석
    print("\n방법 2: 미래 성과 예측력 분석")
    analyze_future_performance(df)
    
    # 방법 3: 현실적 가중치 최적화
    print("\n방법 3: 현실적 가중치 최적화")
    optimize_realistic_weights(df)

def analyze_team_performance(df):
    """팀 성과와 투수 지표의 관계 분석"""
    # 팀 데이터 로드
    try:
        team_df = pd.read_excel('졸프용 데이터베이스.xlsx', sheet_name='팀 성적')
        print(f"   팀 성적 데이터: {team_df.shape}")
        
        # 연도별 팀별 투수 성과 집계
        pitcher_team_stats = df.groupby(['연도', '팀']).agg({
            '보정FIP': 'mean',
            '보정피안타': 'mean',
            '이닝': 'sum'
        }).reset_index()
        
        # 충분한 이닝을 던진 팀만 선택 (연간 1000이닝 이상)
        pitcher_team_stats = pitcher_team_stats[pitcher_team_stats['이닝'] >= 1000]
        
        # 팀 성적과 매칭
        if '승률' in team_df.columns or '승' in team_df.columns:
            merged = pitcher_team_stats.merge(
                team_df, on=['연도', '팀'], how='inner'
            )
            
            if len(merged) > 20:
                analyze_weight_vs_winning(merged)
            else:
                print("   매칭되는 데이터가 부족합니다.")
        else:
            print("   팀 성적 데이터에 승률 정보가 없습니다.")
            
    except Exception as e:
        print(f"   팀 성적 분석 실패: {e}")

def analyze_weight_vs_winning(merged_df):
    """가중치별로 팀 승률 예측력 테스트"""
    print("   팀 승률 예측력 테스트")
    
    # 승률 계산
    if '승률' in merged_df.columns:
        target = merged_df['승률']
    elif '승' in merged_df.columns and '패' in merged_df.columns:
        wins = merged_df['승']
        losses = merged_df['패']
        target = wins / (wins + losses)
    else:
        print("   승률 계산 불가")
        return
    
    # 다양한 가중치 테스트
    weight_combinations = [
        (0.7, 0.3),  # 기존
        (0.6, 0.4),
        (0.8, 0.2),
        (0.5, 0.5),
        (0.9, 0.1)
    ]
    
    results = []
    
    for fip_weight, hits_weight in weight_combinations:
        # 투수 지표 계산 (낮을수록 좋은 지표로 변환)
        pitcher_score = (fip_weight * (100 / merged_df['보정FIP']) + 
                        hits_weight * (100 / merged_df['보정피안타']))
        
        # 승률과의 상관관계
        correlation = pitcher_score.corr(target)
        
        results.append({
            'FIP_weight': fip_weight,
            'Hits_weight': hits_weight,
            'correlation': correlation
        })
        
        print(f"   {fip_weight:.1f}:{hits_weight:.1f} → 상관계수: {correlation:.3f}")
    
    # 최고 성능 가중치 찾기
    best_result = max(results, key=lambda x: abs(x['correlation']))
    print(f"   최적 가중치: {best_result['FIP_weight']:.1f}:{best_result['Hits_weight']:.1f}")
    print(f"   최고 상관계수: {best_result['correlation']:.3f}")

def analyze_future_performance(df):
    """미래 성과 예측력 분석"""
    print("   다음 시즌 성과 예측력 테스트")
    
    # 연속된 시즌 데이터가 있는 투수들 찾기
    pitcher_years = df.groupby('선수명')['연도'].apply(list).reset_index()
    
    future_predictions = []
    
    for _, row in pitcher_years.iterrows():
        pitcher = row['선수명']
        years = sorted(row['연도'])
        
        for i in range(len(years) - 1):
            current_year = years[i]
            next_year = years[i + 1]
            
            # 연속된 년도인지 확인
            if next_year - current_year == 1:
                current_data = df[(df['선수명'] == pitcher) & (df['연도'] == current_year)]
                next_data = df[(df['선수명'] == pitcher) & (df['연도'] == next_year)]
                
                if len(current_data) == 1 and len(next_data) == 1:
                    curr = current_data.iloc[0]
                    next_row = next_data.iloc[0]
                    
                    # 충분한 이닝을 던진 경우만
                    if curr['이닝'] >= 50 and next_row['이닝'] >= 50:
                        future_predictions.append({
                            'pitcher': pitcher,
                            'year': current_year,
                            'current_fip': curr['보정FIP'],
                            'current_hits': curr['보정피안타'],
                            'next_era': next_row['보정ERA'],
                            'next_fip': next_row['보정FIP']
                        })
    
    if len(future_predictions) > 20:
        analyze_predictive_weights(future_predictions)
    else:
        print(f"   연속 시즌 데이터가 부족합니다 ({len(future_predictions)}건)")

def analyze_predictive_weights(predictions):
    """예측력 기반 가중치 분석"""
    pred_df = pd.DataFrame(predictions)
    
    print(f"   분석 대상: {len(pred_df)}명의 투수")
    
    # 다양한 가중치로 다음 시즌 성과 예측
    weight_combinations = [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2), (0.5, 0.5)]
    
    best_correlation = 0
    best_weights = (0.7, 0.3)
    
    for fip_w, hits_w in weight_combinations:
        # 현재 시즌 지표로 다음 시즌 ERA 예측
        current_score = fip_w * pred_df['current_fip'] + hits_w * pred_df['current_hits']
        correlation = current_score.corr(pred_df['next_era'])
        
        print(f"   {fip_w:.1f}:{hits_w:.1f} → 예측 상관계수: {correlation:.3f}")
        
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_weights = (fip_w, hits_w)
    
    print(f"   최고 예측력: {best_weights[0]:.1f}:{best_weights[1]:.1f} (r={best_correlation:.3f})")

def optimize_realistic_weights(df):
    """현실적 범위에서 가중치 최적화"""
    print("   현실적 가중치 범위에서 최적화")
    
    # 충분한 이닝을 던진 투수들만
    qualified_df = df[df['이닝'] >= 50].copy()
    print(f"   분석 대상: {len(qualified_df)}명")
    
    # 여러 성과 지표와의 관계 확인
    if 'ERA*' in qualified_df.columns:
        target = qualified_df['ERA*']
        target_name = 'ERA*'
    else:
        # 보정ERA의 역수를 사용 (높을수록 좋게)
        target = 100 / qualified_df['보정ERA']
        target_name = '보정ERA 역수'
    
    print(f"   타겟 지표: {target_name}")
    
    # FIP 중심의 테스트
    best_r2 = -1
    best_weights = (0.7, 0.3)
    
    for fip_weight in np.arange(0.01, 0.99, 0.01):
        hits_weight = 1 - fip_weight
        
        # 가중 점수 계산 (보정된 값이므로 높을수록 좋음)
        weighted_score = fip_weight * qualified_df['보정FIP'] + hits_weight * qualified_df['보정피안타']
        
        # 타겟과의 관계
        correlation = weighted_score.corr(target)
        r2 = correlation ** 2
        
        print(f"   {fip_weight:.2f}:{hits_weight:.2f} → R² = {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_weights = (fip_weight, hits_weight)
    
    print(f"\n   최적 가중치: {best_weights[0]:.2f}:{best_weights[1]:.2f}")
    print(f"   최고 설명력: R² = {best_r2:.4f}")
    
    # 기존 7:3과 비교
    original_score = 0.7 * qualified_df['보정FIP'] + 0.3 * qualified_df['보정피안타']
    original_r2 = (original_score.corr(target)) ** 2
    
    print(f"\n   비교 결과:")
    print(f"   기존 7:3 방식: R² = {original_r2:.4f}")
    print(f"   최적화 방식: R² = {best_r2:.4f}")
    
    improvement = best_r2 - original_r2
    if improvement > 0.01:
        print(f"   개선도: +{improvement:.4f} (유의미한 개선)")
    elif improvement > 0:
        print(f"   개선도: +{improvement:.4f} (미미한 개선)")
    else:
        print(f"   기존 방식이 더 우수함")
    
    return best_weights, best_r2

if __name__ == "__main__":
    correct_weight_analysis()