import numpy as np
from baseball_utils import load_pitcher_data, get_pitcher_stats, load_batter_data, get_season_avg_batter_ops
import os
import pandas as pd

def normalize_fip(fip):
    if fip is None:
        return 0.5
    score = (150 - fip) / 100  # 50~150 범위를 0.0~1.0으로 변환
    return max(0.0, min(score, 1.0)) #score가 높을수록 좋은 투수

def simulate_matchup(pitcher_stats, batter_stats, n_sim=1000):
    fip = pitcher_stats.get('FIP*')
    batter_ops = batter_stats.get('OPS', 0.7)
   
    fip_score = normalize_fip(fip)
    
    hit_prob = (1 - fip_score + batter_ops) / 2    #fip_score가 높을수록 안타를 맞을 확률이 낮아짐
    hit_prob = min(max(hit_prob, 0), 1)
        
    results = np.random.choice(['hit', 'out'], size=n_sim, p=[hit_prob, 1-hit_prob])
    hit_count = np.sum(results == 'hit')
    out_count = np.sum(results == 'out')
    return {
        'hit': hit_count,
        'out': out_count,
        'hit_rate': round(hit_count / n_sim, 3)
    }

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
        if pitcher_stats.get('FIP*') is None:
            continue

        batter_ops = get_season_avg_batter_ops(batter_df, year)
        batter_stats = {'OPS': batter_ops}
        result = simulate_matchup(pitcher_stats, batter_stats)
        results.append({
            '투수명': pitcher_name,
            '연도': year,
            'FIP*': pitcher_stats.get('FIP*'),
            'ERA*': pitcher_stats.get('ERA*'),
            'RA9*': pitcher_stats.get('RA9*'),
            'K/9': pitcher_stats.get('K/9'),
            'BB/9': pitcher_stats.get('BB/9'),
            'HR/9': pitcher_stats.get('HR/9'),
            '피OPS': pitcher_stats.get('피OPS'),
            '타자OPS(시즌평균)': round(batter_ops, 3) if batter_ops else None,
            '시뮬_안타': result['hit'],
            '시뮬_아웃': result['out'],
            '시뮬_안타율': result['hit_rate']
        })
    result_df = pd.DataFrame(results)
    result_df.to_csv('all_pitchers_simulation_results.csv', index=False, encoding='utf-8-sig')