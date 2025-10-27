import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트가 깨지지 않도록 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def validate_2025_prediction_model():
    """
    2025년 실제 데이터를 사용하여, 기존 모델과 신규 모델의
    예측력을 최종적으로 검증합니다.
    """
    print("="*70)
    print("2025 시즌 데이터 기반 모델 예측력 최종 검증")
    print("="*70)

    try:
        # baseball_utils.py를 통해 ERA* 이상치가 보정된 데이터를 사용한다고 가정
        # '투수보정' 시트가 포함된 엑셀 파일을 로드합니다.
        df = pd.read_excel('졸프용 데이터베이스(2025년 추가).xlsx', sheet_name='투수보정')
    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        return

    # --- 1. 2025년 데이터 필터링 ---
    df_2025 = df[df['연도'] == 2025].copy()

    # 통계적 신뢰도를 위해 최소 이닝 자격 조건 설정
    MIN_INNINGS = 30
    df_2025 = df_2025[df_2025['이닝'] >= MIN_INNINGS]

    if len(df_2025) < 10: # 최소 분석 샘플 수 확인
        print(f"분석할 2025년 데이터가 부족합니다. (시즌 {MIN_INNINGS}이닝 이상 투수 {len(df_2025)}명)")
        return
        
    print(f"2025년 분석 대상: 시즌 {MIN_INNINGS}이닝 이상 투수 {len(df_2025)}명")

    # --- 2. 두 가지 모델로 PAI 점수 계산 ---
    # A그룹 (통제군): 기존 7:3 가중치
    df_2025['PAI_73'] = 0.70 * df_2025['보정FIP'] + 0.30 * df_2025['보정피안타']
    # B그룹 (실험군): 새로운 1:2 가중치 (34:66) 
    df_2025['PAI_46'] = 0.34 * df_2025['보정FIP'] + 0.66 * df_2025['보정피안타']

    # --- 3. 2025년 실제 성과(ERA*)와의 상관관계 비교 (핵심 검증) ---
    print("\n" + "="*70)
    print("2025년 실제 성적(ERA*) 예측력 비교")
    print("="*70)
    
    # ERA*가 0 이하인 비현실적 데이터 제외
    analysis_data = df_2025[df_2025['ERA*'] > 0].copy()

    corr_73 = analysis_data['PAI_73'].corr(analysis_data['ERA*'])
    corr_46 = analysis_data['PAI_46'].corr(analysis_data['ERA*'])
    improvement = corr_46 - corr_73

    print(f"  - 기존 모델 (7:3)의 2025년 예측력 (상관계수) : {corr_73:.4f}")
    print(f"  - 신규 모델 (1:2)의 2025년 예측력 (상관계수) : {corr_46:.4f}")
    
    print("-" * 50)
    if improvement > 0.001:
        print(f"  예측력 개선도: +{improvement:.4f}")
        print(f"  => 결론: 신규 모델이 2025년 시즌을 더 정확하게 예측했습니다!")
    else:
        print("  - 기존 모델의 예측력이 더 높거나 유의미한 차이가 없었습니다.")

    # --- 4. 시각화를 통한 결과 증명 ---
    print("\n" + "="*70)
    print("2025년 예측력 비교 시각화")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # 기존 모델 (7:3) 시각화
    sns.regplot(x='PAI_73', y='ERA*', data=analysis_data, ax=axes[0], line_kws={"color": "red"})
    axes[0].set_title(f'기존 모델(7:3) 예측력\n(상관계수: {corr_73:.4f})', fontsize=14)
    axes[0].set_xlabel("PAI (7:3 가중치)")
    axes[0].grid(True)

    # 신규 모델 (4:6) 시각화
    sns.regplot(x='PAI_46', y='ERA*', data=analysis_data, ax=axes[1], line_kws={"color": "red"})
    axes[1].set_title(f'신규 모델(1:2) 예측력\n(상관계수: {corr_46:.4f})', fontsize=14)
    axes[1].set_xlabel("PAI (1:2 가중치)")
    axes[1].grid(True)
    
    fig.suptitle('2025 시즌 PAI 모델별 실제 성적 예측력 비교', fontsize=20, y=0.98)
    
    output_filename = 'prediction_validation_2025.png'
    plt.savefig(output_filename)
    print(f"  - 예측력 비교 그래프 저장 완료: {output_filename}")


if __name__ == "__main__":
    validate_2025_prediction_model()