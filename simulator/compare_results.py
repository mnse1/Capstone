import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_simulation_results():
    """기존 7:3 가중치와 최적화된 39:61 가중치 결과 비교"""
    print("="*70)
    print("시뮬레이션 결과 비교 분석")
    print("="*70)
    
    # 데이터 로드
    try:
        original = pd.read_csv('all_pitchers_simulation_results.csv')
        optimized = pd.read_csv('pitchers_simulation_optimized_weights.csv')
        print(f"기존 결과 (7:3): {len(original)}명")
        print(f"최적화 결과 (39:61): {len(optimized)}명")
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return
    
    # 공통 투수들만 분석 (투수명, 연도 기준)
    original['key'] = original['투수명'].astype(str) + '_' + original['연도'].astype(str)
    optimized['key'] = optimized['투수명'].astype(str) + '_' + optimized['연도'].astype(str)
    
    common_keys = set(original['key']) & set(optimized['key'])
    print(f"공통 투수 데이터: {len(common_keys)}명")
    
    # 공통 데이터 추출
    orig_common = original[original['key'].isin(common_keys)].set_index('key')
    opt_common = optimized[optimized['key'].isin(common_keys)].set_index('key')
    
    # 정렬하여 일치시키기
    common_keys_sorted = sorted(common_keys)
    orig_common = orig_common.loc[common_keys_sorted]
    opt_common = opt_common.loc[common_keys_sorted]
    
    print("\n=== 주요 지표 비교 ===")
    
    # PAI 비교
    pai_orig = orig_common['PAI(100)']
    pai_opt = opt_common['PAI(100)_optimized']
    pai_diff = pai_opt - pai_orig
    
    print(f"\n1. PAI(100) 비교:")
    print(f"   기존 (7:3) 평균: {pai_orig.mean():.1f}")
    print(f"   최적화 (39:61) 평균: {pai_opt.mean():.1f}")
    print(f"   평균 차이: {pai_diff.mean():+.1f}")
    print(f"   표준편차 - 기존: {pai_orig.std():.1f}, 최적화: {pai_opt.std():.1f}")
    
    # 피안타율 비교
    hit_rate_orig = orig_common['시뮬_피안타율']
    hit_rate_opt = opt_common['시뮬_피안타율_opt']
    hit_rate_diff = hit_rate_opt - hit_rate_orig
    
    print(f"\n2. 시뮬레이션 피안타율 비교:")
    print(f"   기존 (7:3) 평균: {hit_rate_orig.mean():.3f}")
    print(f"   최적화 (39:61) 평균: {hit_rate_opt.mean():.3f}")
    print(f"   평균 차이: {hit_rate_diff.mean():+.3f}")
    print(f"   표준편차 - 기존: {hit_rate_orig.std():.3f}, 최적화: {hit_rate_opt.std():.3f}")
    
    # 상관관계 분석
    correlation_pai = np.corrcoef(pai_orig, pai_opt)[0,1]
    correlation_hit = np.corrcoef(hit_rate_orig, hit_rate_opt)[0,1]
    
    print(f"\n3. 상관관계:")
    print(f"   PAI(100) 상관계수: {correlation_pai:.4f}")
    print(f"   피안타율 상관계수: {correlation_hit:.4f}")
    
    # 극값 비교
    print(f"\n4. 극값 분석:")
    print(f"   PAI(100) 최고값 - 기존: {pai_orig.max():.1f}, 최적화: {pai_opt.max():.1f}")
    print(f"   PAI(100) 최저값 - 기존: {pai_orig.min():.1f}, 최적화: {pai_opt.min():.1f}")
    print(f"   피안타율 최고값 - 기존: {hit_rate_orig.max():.3f}, 최적화: {hit_rate_opt.max():.3f}")
    print(f"   피안타율 최저값 - 기존: {hit_rate_orig.min():.3f}, 최적화: {hit_rate_opt.min():.3f}")
    
    # 개선도 분석
    print(f"\n5. 개별 투수별 변화:")
    pai_improved = (pai_diff > 0).sum()
    pai_worsened = (pai_diff < 0).sum()
    pai_same = (pai_diff == 0).sum()
    
    print(f"   PAI 개선된 투수: {pai_improved}명 ({pai_improved/len(pai_diff)*100:.1f}%)")
    print(f"   PAI 악화된 투수: {pai_worsened}명 ({pai_worsened/len(pai_diff)*100:.1f}%)")
    print(f"   PAI 동일한 투수: {pai_same}명 ({pai_same/len(pai_diff)*100:.1f}%)")
    
    hit_improved = (hit_rate_diff < 0).sum()  # 피안타율은 낮을수록 좋음
    hit_worsened = (hit_rate_diff > 0).sum()
    hit_same = (hit_rate_diff == 0).sum()
    
    print(f"   피안타율 개선된 투수: {hit_improved}명 ({hit_improved/len(hit_rate_diff)*100:.1f}%)")
    print(f"   피안타율 악화된 투수: {hit_worsened}명 ({hit_worsened/len(hit_rate_diff)*100:.1f}%)")
    print(f"   피안타율 동일한 투수: {hit_same}명 ({hit_same/len(hit_rate_diff)*100:.1f}%)")
    
    # 실제 성과와의 관계 분석 (ERA* 기준)
    if 'ERA*' in orig_common.columns:
        era_star = orig_common['ERA*']
        valid_era = ~pd.isna(era_star)
        
        if valid_era.sum() > 10:
            print(f"\n6. 실제 성과(ERA*)와의 관계:")
            corr_orig_era = np.corrcoef(pai_orig[valid_era], era_star[valid_era])[0,1]
            corr_opt_era = np.corrcoef(pai_opt[valid_era], era_star[valid_era])[0,1]
            
            print(f"   기존 PAI vs ERA*: {corr_orig_era:.4f}")
            print(f"   최적화 PAI vs ERA*: {corr_opt_era:.4f}")
            print(f"   예측력 개선: {abs(corr_opt_era) - abs(corr_orig_era):+.4f}")

if __name__ == "__main__":
    compare_simulation_results()