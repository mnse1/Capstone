import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트가 깨지지 않도록 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_merge_data(original_file, optimized_file):
    """두 시뮬레이션 결과 파일을 로드하고 공통 데이터를 병합합니다."""
    try:
        original_df = pd.read_csv(original_file)
        optimized_df = pd.read_csv(optimized_file)
        
        # '사용가중치' 컬럼에서 실제 사용된 가중치 라벨을 동적으로 추출
        weights_label = "Optimized"
        if '사용가중치' in optimized_df.columns and not optimized_df['사용가중치'].empty:
            weights_label = optimized_df['사용가중치'].iloc[0]

        print(f"기존 모델 (7:3) 데이터: {len(original_df)}건")
        print(f"신규 모델 ({weights_label}) 데이터: {len(optimized_df)}건")

        # '투수명'과 '연도'를 기준으로 두 데이터프레임을 병합
        merged_df = pd.merge(
            original_df, optimized_df, on=['투수명', '연도'], suffixes=('_orig', '_opt')
        )
        # 중복된 ERA* 컬럼 정리
        merged_df.rename(columns={'ERA*_orig': 'ERA*'}, inplace=True)
        merged_df.drop(columns=['ERA*_opt'], inplace=True, errors='ignore')

        print(f"공통 분석 대상: {len(merged_df)}건")
        return merged_df, weights_label

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return None, None


def analyze_statistics(df, weights_label):
    """주요 통계 지표를 비교 분석하고 출력합니다."""
    print("\n" + "="*70)
    print("1. 주요 지표 통계 비교")
    print("="*70)

    # PAI(Pitcher Assessment Index) 비교
    pai_orig_mean = df['PAI(100)'].mean()
    pai_opt_mean = df['PAI(100)_optimized'].mean()
    pai_orig_std = df['PAI(100)'].std()
    pai_opt_std = df['PAI(100)_optimized'].std()

    print(f"  [PAI(100) 지표 비교]")
    print(f"  - 평균 (7:3)      : {pai_orig_mean:.2f}")
    print(f"  - 평균 ({weights_label}) : {pai_opt_mean:.2f}")
    print(f"  - 표준편차 (7:3)      : {pai_orig_std:.2f} (변동성)")
    print(f"  - 표준편차 ({weights_label}) : {pai_opt_std:.2f} (변동성)")
    if pai_opt_std < pai_orig_std:
        print("  => 해석: 신규 모델이 투수를 더 안정적인 척도로 평가합니다.")


def analyze_correlation_with_era(df, weights_label):
    """실제 성과(ERA*)와의 상관관계를 분석하여 모델의 예측력을 검증합니다."""
    print("\n" + "="*70)
    print("2. 실제 성과(ERA*)와의 예측력 검증 (핵심)")
    print("="*70)
    
    # ERA*가 0 이하인 비현실적인 데이터 제외
    analysis_data = df[df['ERA*'] > 0].copy()

    corr_orig = analysis_data['PAI(100)'].corr(analysis_data['ERA*'])
    corr_opt = analysis_data['PAI(100)_optimized'].corr(analysis_data['ERA*'])
    improvement = corr_opt - corr_orig

    print(f"  - 기존 모델 PAI vs ERA* 상관계수 : {corr_orig:.4f}")
    print(f"  - 신규 모델 PAI vs ERA* 상관계수 : {corr_opt:.4f}")
    
    print("-" * 50)
    if improvement > 0.001:
        print(f"  예측력 개선도: +{improvement:.4f}")
        print(f"  => 결론: 신규 모델이 실제 투수 성적을 {improvement/corr_orig:.2%} 더 정확하게 예측합니다.")
    else:
        print("  - 예측력에 유의미한 변화가 없거나 기존 모델이 더 우수합니다.")

def visualize_results(df, weights_label):
    """분석 결과를 시각화하여 이미지 파일로 저장합니다."""
    print("\n" + "="*70)
    print("3. 분석 결과 시각화")
    print("="*70)

    # 1. 두 모델의 PAI 점수 분포 비교 (KDE Plot)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['PAI(100)'], label='기존 모델 (7:3)', fill=True)
    sns.kdeplot(df['PAI(100)_optimized'], label=f'신규 모델 ({weights_label})', fill=True)
    plt.title('PAI 점수 분포 비교', fontsize=16)
    plt.xlabel('PAI(100)', fontsize=12)
    plt.ylabel('밀도', fontsize=12)
    plt.legend()
    plt.grid(True)
    dist_filename = 'pai_distribution_comparison.png'
    plt.savefig(dist_filename)
    print(f"  - PAI 분포 비교 그래프 저장 완료: {dist_filename}")

    # 2. 실제 성과(ERA*)와의 관계 시각화 (Regression Plot)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    sns.regplot(x='PAI(100)', y='ERA*', data=df, ax=axes[0], line_kws={"color": "red"})
    axes[0].set_title('기존 모델 PAI vs 실제 성과(ERA*)', fontsize=14)
    axes[0].grid(True)

    sns.regplot(x='PAI(100)_optimized', y='ERA*', data=df, ax=axes[1], line_kws={"color": "red"})
    axes[1].set_title(f'신규 모델 PAI vs 실제 성과(ERA*)', fontsize=14)
    axes[1].grid(True)
    
    fig.suptitle('모델별 예측력 비교 시각화', fontsize=20, y=0.98)
    corr_filename = 'pai_correlation_comparison.png'
    plt.savefig(corr_filename)
    print(f"  - 예측력 비교 그래프 저장 완료: {corr_filename}")


if __name__ == "__main__":
    original_file = 'all_pitchers_simulation_results.csv'
    optimized_file = 'pitchers_simulation_optimized_weights.csv'

    merged_df, weights_label = load_and_merge_data(original_file, optimized_file)

    if merged_df is not None:
        analyze_statistics(merged_df, weights_label)
        analyze_correlation_with_era(merged_df, weights_label)
        visualize_results(merged_df, weights_label)