def ewma_estimate(prev_mean, prev_var, obs, lam):
    if prev_mean is None:
        m_new = obs
        v_new = 0.0
    else:
        m_prev = prev_mean
        m_new = lam * m_prev + (1 - lam) * obs
        v_new = lam * prev_var + (1 - lam) * (obs - m_prev)**2
    
    var = max(v_new, 1e-6)
    alpha_hat = m_new**2 / var
    alpha_hat = max(alpha_hat, 1+1e-6)  # Ensure (alpha_hat-1) is positive
    beta_hat  = m_new / var   
    
    return alpha_hat, beta_hat, m_new, v_new