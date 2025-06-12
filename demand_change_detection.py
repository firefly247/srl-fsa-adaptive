import numpy as np
from scipy.special import gammaln, psi
  
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

def is_change_detected(alpha_hist, beta_hist,
                       ewma_lambda=0.1,
                       window=100,
                       kl_threshold=0.5):
    """
    최신 시점에서만 대칭 KL divergence를 계산해
    분포 변화가 있었는지(True/False) 반환.
    - alpha_hist, beta_hist: 추정 α, β 시퀀스 리스트
    - ewma_lambda: 히스토리 평활 계수
    - window: 비교할 과거 시점 차
    - kl_threshold: 대칭 KL 임계치
    """
    n = len(alpha_hist)
    # 검사할 충분한 데이터가 없으면 변화 아님
    if n < window + 1:
        return False

    # 1) EWMA smoothing
    def ewma_series(series, lam):
        out = np.zeros_like(series)
        out[0] = series[0]
        for t in range(1, len(series)):
            out[t] = lam * out[t-1] + (1 - lam) * series[t]
        return out

    a_smooth = ewma_series(np.array(alpha_hist), ewma_lambda)
    b_smooth = ewma_series(np.array(beta_hist),  ewma_lambda)

    # 2) 최신 시점 t = n-1 과 과거 시점 t-window 만 비교
    ap0, bp0 = a_smooth[-window-1], b_smooth[-window-1]
    ap1, bp1 = a_smooth[-1],          b_smooth[-1]

    kl01 = gamma_kl(ap0, bp0, ap1, bp1)
    kl10 = gamma_kl(ap1, bp1, ap0, bp0)
    sym_kl = 0.5 * (kl01 + kl10)

    return sym_kl > kl_threshold
  
