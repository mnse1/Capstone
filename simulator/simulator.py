import numpy as np
from baseball_utils import load_pitcher_data, get_pitcher_stats, load_batter_data, get_season_avg_batter_ops
import os
import pandas as pd

# 100 스케일(50~150 가정) → 0~1
def normalize_100(value, low=50, high=150):
    if value is None or pd.isna(value):
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))

def simulate_matchup(pitcher_stats, batter_ops, n_sim=1000, w_fip=0.7, w_hit=0.3):
    # 보정 스코어 (높을수록 좋음)
    adj_fip = pitcher_stats.get('보정FIP')
    adj_hit = pitcher_stats.get('보정피안타')

    # 필수 값 결여 시 시뮬 생략
    if adj_fip is None or adj_hit is None:
        return None

    s_fip = normalize_100(adj_fip)
    s_hit = normalize_100(adj_hit)
    combined_score = w_fip * s_fip + w_hit * s_hit  # 0~1

    # 리그 평균 타자 영향은 "연도 평균 OPS"만 사용 (투수 능력 위주)
    # hit 확률 맵핑: 평균 조정(0.7 OPS) 근처에서 합리적 범위로
    # 필요시 a, b 튜닝 가능
    a = 0.30  # 기본 베이스(리그 평균 단타 확률 감각)
    b = 0.25  # 타자 OPS 기여 계수
    base = a + b * (batter_ops - 0.7)

    # combined_score가 높을수록 p_hit 감소
    p_hit = np.clip(base + (0.35 * (1.0 - combined_score)), 0.02, 0.60)

    results = np.random.choice(['hit', 'out'], size=n_sim, p=[p_hit, 1 - p_hit])
    hit_count = int(np.sum(results == 'hit'))
    out_count = n_sim - hit_count

    # 100 스케일 결합지표(PAI)도 같이 산출 (해석 쉬움)
    pai_100 = 100 * (w_fip * (adj_fip / 100.0) + w_hit * (adj_hit / 100.0))

    return {
        'hit': hit_count,
        'out': out_count,
        'hit_rate': round(hit_count / n_sim, 3),
        'PAI(100)': round(pai_100, 1),
        '보정FIP': round(adj_fip, 1) if adj_fip is not None else None,
        '보정피안타': round(adj_hit, 1) if adj_hit is not None else None
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
