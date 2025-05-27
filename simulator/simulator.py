# simulator.py
# \Capstone\simulator> python .\simulator.py to run the simulation

from models import Pitcher, Batter
from match_engine import MatchEngine

def run_simulation(pitcher, batter, num_at_bats=1000):  
    results = {'strikeout': 0, 'walk': 0, 'hit': 0, 'out': 0}
    
    engine = MatchEngine(pitcher, batter)
    
    for _ in range(num_at_bats):
        result = engine.calc_outcome()
        results[result] += 1
    
    return results

# 예제 실행
if __name__ == "__main__":
    # 임시 데이터 (실제 데이터 로드 필요)
    pitcher = Pitcher("류현진", 2024, "KBO", fastball=80, control=75, breaking=70, stamina=85, era_star=100, fip_star=95)
    batter = Batter("이정후", 2024, "KBO", avg=0.320, obp=0.400, slg=0.500, k_rate=0.15, bb_rate=0.12, wrc_plus=140)

    results = run_simulation(pitcher, batter, num_at_bats=10000)
    print("시뮬레이션 결과:", results)
