import numpy as np
import yaml
import matplotlib.pyplot as plt
from main import estimate_gamma_params

def load_regimes(config_path='config.yaml'):
    """
    config.yaml에서 regime_sequence를 로드합니다.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    regimes = cfg.get('regime_sequence', [])
    # 각 항목이 [alpha, beta] 형태인지 확인
    return [(float(a), float(b)) for a, b in regimes]

def plot_history(alpha_history, beta_history):
    """
    alpha_hat 과 beta_hat 의 추정값을 시간의 흐름에 따라 시각화합니다.
    """
    steps = np.arange(1, len(alpha_history) + 1)
    plt.figure(figsize=(10, 5))

    # alpha_hat plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, alpha_history, marker='.', linestyle='-')
    plt.title('Alpha Hat Over Time')
    plt.xlabel('Step')
    plt.ylabel('Alpha Hat')
    plt.grid(True)

    # beta_hat plot
    plt.subplot(1, 2, 2)
    plt.plot(steps, beta_history, marker='.', linestyle='-')
    plt.title('Beta Hat Over Time')
    plt.xlabel('Step')
    plt.ylabel('Beta Hat')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def cusum():
    # 샘플 수 per regime
    samples_per_regime = 1000
    # 초기 워밍업 단계(스텝 수)
    warmup_steps = 100

    # config.yaml에서 레짐 파라미터 로드
    regime_sequence = load_regimes('config.yaml')

    demand_samples = []
    alpha_history = []
    beta_history = []
    step = 0

    # 각 레짐별 샘플링
    for alpha, beta in regime_sequence:
        for _ in range(samples_per_regime):
            step += 1
            # demand ~ Gamma(alpha, rate=beta)
            d = np.random.gamma(shape=alpha, scale=1.0/beta)
            demand_samples.append(max(d,0))

            # 워밍업 후부터 파라미터 추정 및 기록
            if step > warmup_steps:
                a_hat, b_hat = estimate_gamma_params(demand_samples)
                alpha_history.append(a_hat)
                beta_history.append(b_hat)

    # 기록된 스텝 수 출력
    print(f"Total recorded steps: {len(alpha_history)} (warmup {warmup_steps} steps excluded)")

    # 시각화
    plot_history(alpha_history, beta_history)
    
def ewma():
    # EWMA smoothing factor (lambda): 과거 영향력 (0<lambda<1)
    lambda_val = 0.99
    # regime당 샘플 수
    samples_per_regime = 1000
    # 초기 버퍼 단계(기록 미적용 스텝 수)
    buffer_steps = 200

    regime_sequence = load_regimes('config.yaml')

    # 초기 EWMA 값 설정
    m = None  # EWMA mean
    v = None  # EWMA variance
    alpha_history = []
    beta_history = []
    step = 0

    for alpha, beta in regime_sequence:
        for _ in range(samples_per_regime):
            step += 1
            # demand ~ Gamma(alpha_true, rate=beta_true)
            d = np.random.gamma(shape=alpha, scale=1.0/beta)

            # EWMA 업데이트
            if m is None:
                # 첫 샘플 초기화
                m = d
                v = 0.0
            else:
                m_prev = m
                # EWMA mean
                m = lambda_val * m_prev + (1 - lambda_val) * d
                # EWMA variance
                v = lambda_val * v + (1 - lambda_val) * (d - m_prev)**2

            # EWMA 안정화용 버퍼 후에만 기록
            if step > buffer_steps:
                # alpha_hat, beta_hat 계산 (모멘트 기반)
                if v <= 0:
                    a_hat = 1.0
                    b_hat = 1.0
                else:
                    a_hat = m**2 / v
                    b_hat = m / v
                alpha_history.append(a_hat)
                beta_history.append(b_hat)

    print(f"Total recorded steps: {len(alpha_history)} (buffer {buffer_steps} steps excluded)")
    plot_history(alpha_history, beta_history)

if __name__ == "__main__":
    cusum()
    ewma()