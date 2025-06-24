import pandas as pd
import os

def load_pitcher_data(filepath, sheet_name='투수보정'):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def get_pitcher_stats(df, pitcher_name, year):
    pitcher_row = df[(df['선수명'] == pitcher_name) & (df['연도'] == year)]
    if pitcher_row.empty:
        return None
    row = pitcher_row.iloc[0]
    ip = row['이닝']
    so = row.get('삼진', row.get('K', 0))
    bb = row.get('볼넷', None) or row.get('BB', None) or 0
    hr = row.get('피홈런', None) or row.get('HR', None) or 0
    fip = row.get('FIP*', None) or row.get('FIP', None) or None
    era = row.get('ERA*', None) or row.get('ERA', None) or None
    obp_against = row.get('피출루율', None)
    slg_against = row.get('피장타율', None)
    ops_against = None
    if pd.notna(obp_against) and pd.notna(slg_against):
        ops_against = obp_against + slg_against
    k9 = so * 9 / ip if ip else None
    bb9 = bb * 9 / ip if ip else None
    hr9 = hr * 9 / ip if ip else None
    return {
        'K/9': round(k9, 2) if k9 else None,
        'BB/9': round(bb9, 2) if bb9 else None,
        'HR/9': round(hr9, 2) if hr9 else None,
        'FIP': fip,
        'ERA': era,
        '피OPS': round(ops_against, 3) if ops_against else None
    }

def load_batter_data(filepath, sheet_name='타자보정'):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def get_season_avg_batter_ops(batter_df, year):
    year_df = batter_df[batter_df['연도'] == year]
    obp = year_df['출루율'].mean() if '출루율' in year_df.columns else None
    slg = year_df['장타율'].mean() if '장타율' in year_df.columns else None
    if obp is not None and slg is not None:
        return obp + slg
    return 0.7
