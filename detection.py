import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.special import gammaln, psi

def load_regimes(config_path='config.yaml'):
    """
    config.yaml에서 regime_sequence를 로드합니다.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    regimes = cfg.get('regime_sequence', [])
    # 각 항목이 [alpha, beta] 형태인지 확인
    return [(float(a), float(b)) for a, b in regimes]
  
def ewma_estimate(prev_mean, prev_var, obs, lam):
    """
    단일 관측 obs에 대해 EWMA 기반으로 평균과 분산을 업데이트합니다.
    - prev_mean: 이전 EWMA 평균 (None이면 첫 obs로 초기화)
    - prev_var : 이전 EWMA 분산
    - obs      : 새로운 관측값
    - lam      : EWMA 감쇠 계수 (0<lam<1)
    Returns:
      new_mean, new_var
    """
    if prev_mean is None:
        return obs, 0.0
    m_prev = prev_mean
    m_new = lam * m_prev + (1 - lam) * obs
    v_new = lam * prev_var + (1 - lam) * (obs - m_prev)**2
    return m_new, v_new  

def gamma_kl(alpha_p, beta_p, alpha_q, beta_q):
    """
    두 Gamma 분포 P(alpha_p, beta_p)와 Q(alpha_q, beta_q) 간의 KL divergence KL[P||Q]를 계산합니다.
    rate parameterization: f(x)=β^α/Γ(α) x^{α-1} e^{-βx}
    """
    # KL(P||Q) = α_p*log(β_p/β_q) - logΓ(α_p) + logΓ(α_q)
    #            + (α_p - α_q) ψ(α_p)
    #            + α_q*(β_p/β_q - 1)
    return (
        alpha_p * np.log(beta_p / beta_q)
        - gammaln(alpha_p) + gammaln(alpha_q)
        + (alpha_p - alpha_q) * psi(alpha_p)
        + alpha_q * (beta_p / beta_q - 1)
    )
    
def detect_distribution_changes(alpha_hist, beta_hist,
                                ewma_lambda=0.1,
                                window=100,
                                kl_threshold=0.5):
    """
    KL divergence 기반의 분포 변화 지점 탐지.
    - alpha_hist, beta_hist: 모멘트법 또는 EWMA 추정 α, β 시퀀스
    - ewma_lambda: 감쇠 계수 (EWMA smoothing for estimate series)
    - window: 비교 시차
    - kl_threshold: 대칭 KL 임계치

    Returns:
      change_points: 변화로 판단된 인덱스 리스트
    """
    # 1) EWMA smoothing on the history
    def ewma_series(series, lam):
        out = np.zeros_like(series)
        out[0] = series[0]
        for t in range(1, len(series)):
            out[t] = lam * out[t-1] + (1 - lam) * series[t]
        return out

    a_smooth = ewma_series(np.array(alpha_hist), ewma_lambda)
    b_smooth = ewma_series(np.array(beta_hist),  ewma_lambda)
    change_points = []
    # 2) sliding window symmetric KL
    for t in range(window, len(a_smooth)):
        ap0, bp0 = a_smooth[t - window], b_smooth[t - window]
        ap1, bp1 = a_smooth[t],         b_smooth[t]
        kl01 = gamma_kl(ap0, bp0, ap1, bp1)
        kl10 = gamma_kl(ap1, bp1, ap0, bp0)
        sym_kl = 0.5 * (kl01 + kl10)
        if sym_kl > kl_threshold:
            change_points.append(t)
    return change_points
    
def window_kl():
    # EWMA factor
    lambda_val = 0.99
    samples_per_regime = 1000
    buffer_steps = 200
    regime_sequence = load_regimes('config.yaml')

    m = None
    v = None
    alpha_history = []
    beta_history = []
    step = 0

    for alpha, beta in regime_sequence:
        for _ in range(samples_per_regime):
            step += 1
            d = np.random.gamma(shape=alpha, scale=1.0/beta)

            # EWMA estimation
            m, v = ewma_estimate(m, v, d, lambda_val)
            if step > buffer_steps:
                if v <= 0:
                    a_hat, b_hat = 1.0, 1.0
                else:
                    a_hat, b_hat = m**2 / v, m / v
                alpha_history.append(a_hat)
                beta_history.append(b_hat)

    # 변화 탐지
    change_points = detect_distribution_changes(
        alpha_history, beta_history,
        ewma_lambda=0.5,
        window=100,
        kl_threshold=1.0
    )
    print("Detected change points at steps:", change_points)

    # 시각화
    plot_history(alpha_history, beta_history)
    
def plot_history(alpha_history, beta_history):
    """시각화"""
    steps = np.arange(1, len(alpha_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps, alpha_history, marker='.', linestyle='-')
    plt.title('EWMA Alpha Hat Over Time')
    plt.xlabel('Step')
    plt.ylabel('Alpha Hat')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(steps, beta_history, marker='.', linestyle='-')
    plt.title('EWMA Beta Hat Over Time')
    plt.xlabel('Step')
    plt.ylabel('Beta Hat')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    window_kl()