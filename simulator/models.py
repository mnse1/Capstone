# models.py

class Pitcher:
    def __init__(self, name, year, team, fastball, control, breaking, stamina, era_star, fip_star):
        self.name = name
        self.year = year
        self.team = team
        self.fastball = fastball      # 구위
        self.control = control        # 제구
        self.breaking = breaking      # 변화구
        self.stamina = stamina        # 체력
        self.era_star = era_star      # ERA*
        self.fip_star = fip_star      # FIP*
    
    def get_ability_score(self):
        # 기본 능력치 평균 (예시)
        return (self.fastball + self.control + self.breaking) / 3


class Batter:
    def __init__(self, name, year, team, avg, obp, slg, k_rate, bb_rate, wrc_plus):
        self.name = name
        self.year = year
        self.team = team
        self.avg = avg            # 타율
        self.obp = obp            # 출루율
        self.slg = slg            # 장타율
        self.k_rate = k_rate      # 삼진율
        self.bb_rate = bb_rate    # 볼넷율
        self.wrc_plus = wrc_plus  # wRC+
    
    def get_hitting_score(self):
        # 기본 타격 능력치 (예시)
        return (self.avg + self.obp + self.slg) / 3
