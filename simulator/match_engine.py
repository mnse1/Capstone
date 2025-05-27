# match_engine.py

import random

class MatchEngine:
    def __init__(self, pitcher, batter):
        self.pitcher = pitcher
        self.batter = batter
    
    def calc_outcome(self):
        # 볼넷 확률: 제구력이 높을수록 감소 (최대 50% 감소)
        bb_prob = self.batter.bb_rate - (self.pitcher.control / 100) * self.batter.bb_rate * 0.5
        bb_prob = max(0, bb_prob)

        # 삼진 확률: 제구력이 높을수록 증가 (최대 50% 증가)
        k_prob = self.batter.k_rate + (self.pitcher.control / 100) * self.batter.k_rate * 0.5
        k_prob = max(0, k_prob)

        # 안타 확률: 타자의 타율 사용
        hit_prob = self.batter.avg
        hit_prob = max(0, hit_prob)

        # 나머지는 아웃
        out_prob = max(0, 1 - (bb_prob + k_prob + hit_prob))
        
        # 확률 합이 1이 되도록 정규화
        total = bb_prob + k_prob + hit_prob + out_prob
        if total > 0:
            bb_prob /= total
            k_prob /= total
            hit_prob /= total
            out_prob /= total

        outcomes = ['walk', 'strikeout', 'hit', 'out']
        probabilities = [bb_prob, k_prob, hit_prob, out_prob]

        result = random.choices(outcomes, probabilities)[0]
        return result