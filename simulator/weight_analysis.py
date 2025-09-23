import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def correct_weight_analysis():
    """올바른 접근법: 투수 능력 지표 최적화"""
    print("투수 능력 지표 최적화 분석 (전체 범위 1%~99%)")
    print("="*70)
    
    # 데이터 로드
    file_path = '졸프용 데이터베이스.xlsx'
    df = pd.read_excel(file_path, sheet_name='투수보정')
    
    print(f"전체 데이터: {df.shape}")
    
    # 결과 저장용
    results = {}
    
    # 방법 1: 팀 성과와의 관계 분석
    print("\n" + "="*70)
    print("방법 1: 팀 승률과의 관계 분석")
    print("="*70)
    team_result = analyze_team_performance(df)
    if team_result:
        results['team_performance'] = team_result
    
    # 방법 2: 미래 성과 예측력 분석
    print("\n" + "="*70)
    print("방법 2: 미래 성과 예측력 분석")
    print("="*70)
    future_result = analyze_future_performance(df)
    if future_result:
        results['future_prediction'] = future_result
    
    # 방법 3: 현실적 가중치 최적화
    print("\n" + "="*70)
    print("방법 3: 개별 투수 성과 최적화")
    print("="*70)
    realistic_result = optimize_realistic_weights(df)
    if realistic_result:
        results['individual_performance'] = realistic_result
    
    # 종합 결과 분석
    print("\n" + "="*70)
    print("종합 결과 분석")
    print("="*70)
    
    if results:
        print("각 방법별 최적 가중치:")
        for method, (weights, score) in results.items():
            method_names = {
                'team_performance': '팀 승률 예측',
                'future_prediction': '미래 성과 예측', 
                'individual_performance': '개별 성과 분석'
            }
            fip_w, hits_w = weights
            print(f"   {method_names[method]:12s}: {fip_w:.2f}:{hits_w:.2f} (성능: {score:.4f})")
        
        # 평균 가중치 계산
        all_fip_weights = [w[0] for w, s in results.values()]
        all_hits_weights = [w[1] for w, s in results.values()]
        
        avg_fip = np.mean(all_fip_weights)
        avg_hits = np.mean(all_hits_weights)
        
        print(f"\n종합 권장 가중치: {avg_fip:.2f}:{avg_hits:.2f}")
        print(f"기존 7:3 방식과 비교: {0.7-avg_fip:+.2f}p (FIP), {0.3-avg_hits:+.2f}p (피안타)")
        
        # 결론
        if avg_hits > 0.5:
            print(f"\n결론: 피안타 중심 평가가 더 효과적")
            print(f"   전통적 FIP 중심(70%) → 데이터 기반 피안타 중심({avg_hits:.0%})")
        elif abs(avg_fip - 0.5) < 0.1:
            print(f"\n결론: FIP와 피안타의 균형잡힌 조합이 최적")
            print(f"   50:50 균형 방식 권장")
        else:
            print(f"\n결론: FIP 중심이지만 기존보다 피안타 비중 증가")
    
    return results

def analyze_team_performance(df):
    """팀 성과와 투수 지표의 관계 분석"""
    # 팀 데이터 로드
    try:
        team_df = pd.read_excel('졸프용 데이터베이스.xlsx', sheet_name='팀 성적')
        print(f"팀 성적 데이터: {team_df.shape}")
        
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
                return analyze_weight_vs_winning(merged)
            else:
                print("   매칭되는 데이터가 부족합니다.")
                return None
        else:
            print("   팀 성적 데이터에 승률 정보가 없습니다.")
            return None
            
    except Exception as e:
        print(f"   팀 성적 분석 실패: {e}")
        return None

def analyze_weight_vs_winning(merged_df):
    """가중치별로 팀 승률 예측력 테스트"""
    print("   팀 승률 예측력 테스트 (1%~99% 전체 범위)")
    
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
    
    # 전체 범위 가중치 테스트 (1%~99%)
    best_correlation = 0
    best_weights = (0.7, 0.3)
    results = []
    
    for fip_weight in np.arange(0.01, 0.99, 0.01):
        hits_weight = 1 - fip_weight
        
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
        
        print(f"   {fip_weight:.2f}:{hits_weight:.2f} → 상관계수: {correlation:.4f}")
        
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_weights = (fip_weight, hits_weight)
    
    # 최고 성능 가중치 찾기
    print(f"\n   최적 가중치: {best_weights[0]:.2f}:{best_weights[1]:.2f}")
    print(f"   최고 상관계수: {best_correlation:.4f}")
    
    # 기존 방식과 비교
    original_score = (0.7 * (100 / merged_df['보정FIP']) + 0.3 * (100 / merged_df['보정피안타']))
    original_corr = original_score.corr(target)
    
    print(f"   기존 7:3 방식: {original_corr:.4f}")
    improvement = abs(best_correlation) - abs(original_corr)
    print(f"   개선도: {improvement:+.4f}")
    
    return best_weights, best_correlation

def analyze_future_performance(df):
    """미래 성과 예측력 분석"""
    print("다음 시즌 성과 예측력 테스트")
    
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
        return analyze_predictive_weights(future_predictions)
    else:
        print(f"   연속 시즌 데이터가 부족합니다 ({len(future_predictions)}건)")
        return None

def analyze_predictive_weights(predictions):
    """예측력 기반 가중치 분석"""
    pred_df = pd.DataFrame(predictions)
    
    print(f"   분석 대상: {len(pred_df)}명의 투수")
    print("   미래 성과 예측력 테스트 (1%~99% 전체 범위)")
    
    # 전체 범위 가중치로 다음 시즌 성과 예측
    best_correlation = 0
    best_weights = (0.7, 0.3)
    
    for fip_weight in np.arange(0.01, 0.99, 0.01):
        hits_weight = 1 - fip_weight
        
        # 현재 시즌 지표로 다음 시즌 ERA 예측
        current_score = fip_weight * pred_df['current_fip'] + hits_weight * pred_df['current_hits']
        correlation = current_score.corr(pred_df['next_era'])
        
        print(f"   {fip_weight:.2f}:{hits_weight:.2f} → 예측 상관계수: {correlation:.4f}")
        
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_weights = (fip_weight, hits_weight)
    
    print(f"\n   최고 예측력: {best_weights[0]:.2f}:{best_weights[1]:.2f} (r={best_correlation:.4f})")
    
    # 기존 방식과 비교
    original_score = 0.7 * pred_df['current_fip'] + 0.3 * pred_df['current_hits']
    original_corr = original_score.corr(pred_df['next_era'])
    
    print(f"   기존 7:3 방식: {original_corr:.4f}")
    improvement = abs(best_correlation) - abs(original_corr)
    print(f"   개선도: {improvement:+.4f}")
    
    return best_weights, best_correlation

def optimize_realistic_weights(df):
    """현실적 범위에서 가중치 최적화"""
    print("   현실적 가중치 범위에서 최적화 (1%~99% 전체 범위)")
    
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
    
    # 전체 범위 테스트
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