import pandas as pd

def load_pitcher_data(filepath, sheet_name='투수보정'):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    zero_era_indices = df[df['ERA'] == 0].index
    if not zero_era_indices.empty:
        df.loc[zero_era_indices, 'ERA*'] = 100
    return df

def _safe_mean(series):
    s = pd.to_numeric(series, errors='coerce')
    return s.mean() if s.notna().any() else None

def get_pitcher_stats(df, pitcher_name, year):
    # 해당 투수 행
    row_q = df[(df['선수명'] == pitcher_name) & (df['연도'] == year)]
    if row_q.empty:
        return None
    row = row_q.iloc[0]

    # 연도별 서브셋
    year_df = df[df['연도'] == year]

    # 기본값들
    ip  = row.get('이닝', None)
    so  = row.get('삼진', None)
    bb  = row.get('볼넷', None)
    hr  = row.get('피홈런', None)

    # 리그 평균(연도별) FIP* / 피안타 계산
    league_fip_star = _safe_mean(year_df.get('FIP*'))  # FIP*가 있음을 가정
    league_hits     = _safe_mean(year_df.get('피안타')) if '피안타' in year_df else None

    # 보정FIP 읽기 또는 계산(100 * league / player)
    adj_fip = row.get('보정FIP', None)
    if pd.isna(adj_fip) or adj_fip is None:
        fip_star = row.get('FIP*', None)
        if league_fip_star and fip_star:
            adj_fip = 100 * (league_fip_star / (fip_star + 1e-9))
        else:
            adj_fip = None

    # 보정피안타 읽기 또는 계산(가능하면 H 기준, 없으면 None)
    adj_hits = row.get('보정피안타', None)
    if (pd.isna(adj_hits) or adj_hits is None) and ('피안타' in df.columns):
        h_player = row.get('피안타', None)
        if league_hits and h_player:
            adj_hits = 100 * (league_hits / (h_player + 1e-9))

    # 피출루율/피장타율로 피OPS
    obp_against = row.get('피출루율', None)
    slg_against = row.get('피장타율', None)
    ops_against = (obp_against + slg_against) if pd.notna(obp_against) and pd.notna(slg_against) else None

    # Rate들
    def rate9(x):
        return round(x * 9 / ip, 2) if ip and pd.notna(x) else None

    return {
        'K/9': rate9(so),
        'BB/9': rate9(bb),
        'HR/9': rate9(hr),
        'FIP*': row.get('FIP*', None),
        'ERA*': row.get('ERA*', None),
        'RA9*': row.get('RA9*', None),
        '피OPS': round(ops_against, 3) if ops_against else None,
        '보정FIP': adj_fip,
        '보정피안타': adj_hits
    }

def load_batter_data(filepath, sheet_name='타자보정'):
    return pd.read_excel(filepath, sheet_name=sheet_name)

def get_season_avg_batter_ops(batter_df, year):
    year_df = batter_df[batter_df['연도'] == year]
    obp = year_df['출루율'].mean() if '출루율' in year_df.columns else None
    slg = year_df['장타율'].mean() if '장타율' in year_df.columns else None
    return (obp + slg) if (obp is not None and slg is not None) else 0.7
