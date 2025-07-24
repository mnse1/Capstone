import numpy as np
from baseball_utils import load_pitcher_data, get_pitcher_stats, load_batter_data, get_season_avg_batter_ops
import os
import pandas as pd

def simulate_matchup(pitcher_stats, batter_stats, n_sim=1000):
    # 피OPS, OPS 값이 None일 경우 기본값(0.7)으로 대체
    pitcher_ops = pitcher_stats.get('피OPS', 0.7)
    if pitcher_ops is None:
        pitcher_ops = 0.7
    batter_ops = batter_stats.get('OPS', 0.7) if batter_stats else 0.7
    if batter_ops is None:
        batter_ops = 0.7
    hit_prob = (pitcher_ops + batter_ops) / 2
    hit_prob = min(max(hit_prob, 0), 1)  # 0~1 사이로 보정
    results = np.random.choice(['hit', 'out'], size=n_sim, p=[hit_prob, 1-hit_prob])
    hit_count = np.sum(results == 'hit')
    out_count = np.sum(results == 'out')
    return {'hit': hit_count, 'out': out_count, 'hit_rate': hit_count/n_sim}

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), '졸프용 데이터베이스.xlsx')
    df = load_pitcher_data(file_path)
    batter_df = load_batter_data(file_path)
    results = []
    for idx, row in df.iterrows():
        pitcher_name = row['선수명']
        year = row['연도']
        pitcher_stats = get_pitcher_stats(df, pitcher_name, year)
        if pitcher_stats is None:
            continue
        batter_ops = get_season_avg_batter_ops(batter_df, year)
        batter_stats = {'OPS': batter_ops}
        result = simulate_matchup(pitcher_stats, batter_stats)
        results.append({
            '투수명': pitcher_name,
            '연도': year,
            **pitcher_stats,
            '타자OPS(시즌평균)': round(batter_ops, 3),
            **result
        })
    result_df = pd.DataFrame(results)
    result_df.to_csv('all_pitchers_simulation_results.csv', index=False, encoding='utf-8-sig')