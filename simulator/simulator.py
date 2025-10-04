import numpy as np
from baseball_utils import load_pitcher_data, get_pitcher_stats, load_batter_data, get_season_avg_batter_ops
import os
import pandas as pd

# 100 스케일(50~150 가정) → 0~1
def normalize_100(value, low=50, high=150):
    if value is None or pd.isna(value):
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))

class SimulationConfig:
    # KBO 최근 5년 평균 타율을 기준으로 한 기본 안타 확률
    BASE_HIT_RATE = 0.275
    
    # 타자 OPS가 안타 확률에 미치는 영향 계수 (기존 0.25 -> 0.15로 완화)
    # OPS 0.1 상승 시, 안타 확률 0.015 상승
    OPS_COEFFICIENT = 0.15
    
    # 투수 능력이 안타 확률에 미치는 최대 영향력 (기존 0.35 -> 0.25로 완화)
    # 최고 투수와 최악 투수의 안타 확률 차이를 최대 2할 5푼으로 제한
    PITCHER_MAX_IMPACT = 0.25
    
    # 확률의 현실적 범위 제한
    MIN_HIT_RATE = 0.150  # 아무리 좋은 투수라도 최소 1할 5푼의 안타는 허용
    MAX_HIT_RATE = 0.450  # 아무리 나쁜 투수라도 4할 5푼 이상의 안타는 비현실적

def simulate_matchup(pitcher_stats, batter_ops, n_sim=1000, w_fip=0.7, w_hit=0.3):
    """
    개선된 매개변수와 확률 범위 제한을 적용하여
    투수와 타자의 맞대결을 시뮬레이션합니다.
    """
    # ... (기존 코드: ps, combined_score 계산 등은 동일)
    ps = pitcher_stats
    combined_score = (w_fip * ps['보정FIP'] + w_hit * ps['보정피안타']) / 100
    
    # --- 3. 개선된 공식 적용 ---
    # KBO 평균 타율(0.275)을 기준으로 타자 OPS를 반영하여 기본 안타 확률 계산
    base = SimulationConfig.BASE_HIT_RATE + \
           SimulationConfig.OPS_COEFFICIENT * (batter_ops - 0.7)
           
    # 투수 능력(combined_score)을 반영하여 최종 안타 확률 계산
    # (1 - combined_score)는 평균 대비 투수 능력 편차를 의미
    p_hit = base - (SimulationConfig.PITCHER_MAX_IMPACT * (combined_score - 1.0))
    
    # --- 4. 확률 범위 제한 (np.clip) ---
    # 계산된 p_hit 값이 비현실적인 범위를 벗어나지 않도록 강제 조정
    p_hit = np.clip(p_hit, SimulationConfig.MIN_HIT_RATE, SimulationConfig.MAX_HIT_RATE)

    # ... (이하 시뮬레이션 로직은 기존과 동일)
    n_hit = np.random.binomial(n_sim, p_hit)
    n_out = n_sim - n_hit
    hit_rate = n_hit / n_sim if n_sim > 0 else 0

    return {
        '보정FIP': ps['보정FIP'],
        '보정피안타': ps['보정피안타'],
        'hit': n_hit,
        'out': n_out,
        'hit_rate': hit_rate,
        'PAI(100)': (w_fip * ps['보정FIP'] + w_hit * ps['보정피안타'])
    }

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '졸프용 데이터베이스.xlsx')
    df = load_pitcher_data(file_path)
    batter_df = load_batter_data(file_path)

    results = []
    for _, row in df.iterrows():
        name = row['선수명']; year = row['연도']
        ps = get_pitcher_stats(df, name, year)
        if not ps:
            continue

        # 연도 평균 OPS (투수 능력 위주라 타자 영향 최소화)
        batter_ops = get_season_avg_batter_ops(batter_df, year)

        sim = simulate_matchup(ps, batter_ops, n_sim=1000, w_fip=0.7, w_hit=0.3)
        if sim is None:
            continue

        results.append({
            '투수명': name, '연도': year,
            'K/9': ps.get('K/9'), 'BB/9': ps.get('BB/9'), 'HR/9': ps.get('HR/9'),
            'FIP*': ps.get('FIP*'), 'ERA*': ps.get('ERA*'), 'RA9*': ps.get('RA9*'),
            '피OPS': ps.get('피OPS'), '타자OPS(연도평균)': round(batter_ops, 3) if batter_ops else None,
            '보정FIP': sim['보정FIP'], '보정피안타': sim['보정피안타'],
            'PAI(100)': sim['PAI(100)'],
            '시뮬_안타': sim['hit'], '시뮬_아웃': sim['out'], '시뮬_피안타율': sim['hit_rate'],
        })

    if results:
        out = pd.DataFrame(results)
        out.to_csv('all_pitchers_simulation_results.csv', index=False, encoding='utf-8-sig')
        print('Saved: all_pitchers_simulation_results.csv')
    else:
        print('결과가 비어 있습니다. 보정 컬럼 유무를 확인하세요.')
