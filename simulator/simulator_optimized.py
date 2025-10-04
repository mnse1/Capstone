from simulator import simulate_matchup
from baseball_utils import load_pitcher_data, get_pitcher_stats, load_batter_data, get_season_avg_batter_ops
import os
import pandas as pd
from weight_analysis import calculate_optimal_weights

if __name__ == "__main__":
    print("="*70)
    print("데이터 분석을 통해 최적 가중치를 계산합니다...")
    optimal_fip_weight, optimal_hit_weight = calculate_optimal_weights()
    print("="*70)
    
    if optimal_fip_weight is None:
        print("최적 가중치를 계산하지 못해 시뮬레이션을 중단합니다.")
    else:
        print(f"\n2. 계산된 최적 가중치 (FIP {optimal_fip_weight:.2f} : 피안타 {optimal_hit_weight:.2f})로 시뮬레이션을 시작합니다...")
        print("="*70)
        
        file_path = os.path.join(os.path.dirname(__file__), '졸프용 데이터베이스.xlsx')
        df = load_pitcher_data(file_path)
        batter_df = load_batter_data(file_path)

        results = []
        processed_count = 0
        
        for _, row in df.iterrows():
            name = row['선수명']; year = row['연도']
            ps = get_pitcher_stats(df, name, year)
            if not ps:
                continue

            # 연도 평균 OPS (투수 능력 위주라 타자 영향 최소화)
            batter_ops = get_season_avg_batter_ops(batter_df, year)

            # 최적화된 가중치로 시뮬레이션 (기존 함수 재사용)
            sim = simulate_matchup(ps, batter_ops, n_sim=1000, w_fip=optimal_fip_weight, w_hit=optimal_hit_weight)
            if sim is None:
                continue

            results.append({
                '투수명': name, '연도': year,
                'K/9': ps.get('K/9'), 'BB/9': ps.get('BB/9'), 'HR/9': ps.get('HR/9'),
                'FIP*': ps.get('FIP*'), 'ERA*': ps.get('ERA*'), 'RA9*': ps.get('RA9*'),
                '피OPS': ps.get('피OPS'), '타자OPS(연도평균)': round(batter_ops, 3) if batter_ops else None,
                '보정FIP': sim['보정FIP'], '보정피안타': sim['보정피안타'],
                'PAI(100)_optimized': sim['PAI(100)'],
                '시뮬_안타_opt': sim['hit'], '시뮬_아웃_opt': sim['out'], '시뮬_피안타율_opt': sim['hit_rate'],
                '사용가중치': f"FIP:{optimal_fip_weight:.2f},피안타:{optimal_hit_weight:.2f}"
            })
            
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"처리 완료: {processed_count}명")

        if results:
                out = pd.DataFrame(results)
                output_file = 'pitchers_simulation_optimized_weights.csv'
                out.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"\n시뮬레이션 완료, 결과가 {output_file} 파일에 저장되었습니다.")