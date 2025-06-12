import numpy as np
from loguru import logger
import yaml
from inventory_env import InventoryEnv
import matplotlib.pyplot as plt

def phi(x, c):
    x = x / c
    return np.array([x, x**2, x**3, x**4])

def sigmoid(x, t, tau, clip_range=30):
    z = x / tau(t)
    z = np.clip(z, -clip_range, clip_range)  # clip to avoid overflow
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x, t, tau):
    s = sigmoid(x, t, tau)
    return (1 / tau(t)) * s * (1 - s)

def estimate_gamma_params(samples):
    if len(samples) < 2:
        return 2.0, 1.0
    mean = np.mean(samples)
    var = np.var(samples)
    alpha_hat = mean**2 / var if var > 0 else 2.0
    beta_hat = mean / var if var > 0 else 1.0
    return alpha_hat, beta_hat

def update_value_function(w, rho, x, x_next, reward, gamma1, gamma2, t, phi_scale):
    V_x = w @ phi(x, phi_scale)
    V_x_next = w @ phi(x_next, phi_scale)
    delta = reward - rho + V_x_next - V_x

    w += gamma1(t) * delta * phi(x, phi_scale)
    rho += gamma2 * (reward + V_x_next - V_x - rho)

    return w, rho, V_x, V_x_next, delta

def update_policy_parameters(s, S, x, x_next, w, b1, b2, t, alpha_hat, beta_hat, tau, S_tilde, phi_scale, max_inventory):

    # Sampling Bernoulli variables (η_S and η_s)
    eta_S = np.random.binomial(1, 0.5)
    eta_s = np.random.binomial(1, 0.5)

    # Sampling z_s and z_S based on transition model
    # z_S는 S를 조금 바꾸었다면 다음 상태 x'가 어떻게 나왔을지 추정하는 가상의 x' (x'는 실제 trajectory에서 관측된 상태)
    # dP(x'|S_tilde)/dS를 직접 구하기 어려우므로 샘플 기반으로 근사
    if eta_S == 0: # S는 주문이 발생했을 때만 영향을 미치는 파라미터, 주문을 안했으면 S가 사용되지도 않고 학습할 필요도 없으므로 주문이 일어난 경우의 전이확률만 고려
        z_S = S_tilde - np.random.gamma(alpha_hat-1, 1 / beta_hat)
    else:
        z_S = S_tilde - np.random.gamma(alpha_hat, 1 / beta_hat)

    if eta_s == 0: # s는 주문 여부를 결정하는 임계점으로 행동을 하지 않은 경우/한 경우 모두 반영 필요
        z_s = x - np.random.gamma(alpha_hat-1, 1 / beta_hat) # 주문 안한 경우
    else:
        z_s = S_tilde - np.random.gamma(alpha_hat, 1 / beta_hat) # 주문한 경우

    # s를 조금 바꾸었을 때 그것이 x → x'로 갈 확률을 얼마나 바꾸는지 계산
    # logger.info(f"[{t}] z_s: {z_s:.4f}, z_S: {z_S:.4f}, eta_s: {eta_s}, eta_S: {eta_S}")

    # Compute policy function f(x, s) and its derivative
    f_xs = sigmoid(x - s, t, tau)
    df_dy = sigmoid_derivative(x - s, t, tau)

    # Get value estimates from critic
    V_zs = w @ phi(z_s, phi_scale)
    V_zS = w @ phi(z_S, phi_scale)

    # Apply policy update rules as per SRL-FSA
    S -= b1(t) * beta_hat * (1 - f_xs) * ((-1) ** eta_S) * V_zS
    s -= b2(t) * df_dy * ((-1) ** eta_s) * V_zs

    # logger.info(f"[{t}] S: {S:.4f}, s: {s:.4f}, "
    #             f"b1(t): {b1(t):.4f}, b2(t): {b2(t):.4f}, "
    #             f"f_xs: {f_xs:.4f}, df_dy: {df_dy:.4f}, "
    #             f"eta_S: {eta_S}, eta_s: {eta_s}, "
    #             f"V_zs: {V_zs:.4f}, V_zS: {V_zS:.4f}, ")

    S = np.clip(S, -np.inf, max_inventory)  # Ensure S <= max_inventory
    s = np.clip(s, -np.inf, S)  # Ensure s <= S

    return s, S, V_zs, V_zS

def train_agent(env, config):
    
    # config에서 초기 파라미터 불러오기
    phi_scale = config.get("phi_scale", 50)
    max_inventory = config.get("max_inventory", 1000)
    s = config["s_init"]
    S = config["S_init"]
    rho = config["rho_init"]

    b1 = lambda t: 4 / (np.floor(t/10) + 1)
    b2 = lambda t: 10 / (np.floor(t/20) + 1)
    gamma1 = lambda t: 0.1 / (t + 1) ** 0.7
    gamma2 = 0.01

    tau = lambda t: config["tau_init"] / (np.floor(t/10) + 1) ** 0.8
    sigma = lambda t: config["sigma_init"] / (np.floor(t/10) + 1) ** 0.8
    
    warmup_period = config.get("warmup_period", 60)
    train_period = config.get("train_period", 1000)
    episodes = warmup_period + len(config["regime_sequence"]) * (train_period)

    w = np.zeros(4)
    demand_history = []
    cost_history = []

    obs = env.reset()
    x = obs[0]

    for t in range(episodes):
        # step 2 : observe the transitioned state and corresponding reward after taking action at given state x_t
        # 정책 기반 행동 선택
        # 현재 재고 x가 reorder point s보다 작은지 soft하게 판별
        # deterministic (𝑠, 𝑆) 정책은 MDP에서 단일 커뮤니케이션 클래스를 만들지 않기 때문에 → 일반적인 RL convergence 조건을 만족하지 않음
        # 그래서 noise와 soft decision을 추가해서 → 모든 상태 reachable하게 만듦
        prob = sigmoid(x - s, t, tau)

        S_tilde = S
        if np.random.rand() > prob: # 확률적으로 주문 여부 결정
            noise = np.random.normal(0, sigma(t))
            S_tilde = S + noise
            a = max(S_tilde - x, 0) # 약간의 noise를 추가한 S tilde까지 주문
        else:
            a = 0 # 주문 안함

        # 환경 상호작용
        d, x_next, reward = env.step(a)

        # step 3 : attain realized demand and adaptively estimate the distributional parameters
        demand_history.append(max(d, 0))  # 수요는 비음수
        cost_history.append(-reward)
        alpha_hat, beta_hat = estimate_gamma_params(demand_history)
        # logger.info(f"[{t}] Demand: {d:.4f}, alpha={alpha_hat:.4f}, beta={beta_hat:.4f}, x={x:.4f}, x_next={x_next:.4f}, a={a:.4f}")

        if t >= warmup_period:
            # step 4 : update the relative value function
            w, rho, V_x, V_x_next, delta = update_value_function(w, rho, x, x_next, reward, gamma1, gamma2, t, phi_scale)
            
            rho_history.append(rho)
            w_history.append(w)

            # logger.info(
            #     f"[{t}] "
            #     f"w={w}, "
            #     f"rho={rho:.4f}, "
            #     f"V_x={V_x:.4f}, V_x_next={V_x_next:.4f}, "
            #     f"delta={delta:.4f}, "
            # )

            # step 5 : update the policy parameters
            s, S, V_zs, V_zS = update_policy_parameters(s, S, x, x_next, w, b1, b2, t, alpha_hat, beta_hat, tau, S_tilde, phi_scale, max_inventory)

            if t % 100 == 0:
                logger.info(
                    f"[{t}] "
                    f"s={s:.4f}, S={S:.4f}, "
                    f"V_zs={V_zs:.4f}, V_zS={V_zS:.4f}, "
                    f"alpha_hat={alpha_hat:.4f}, beta_hat={beta_hat:.4f}, "
                )

            s_history.append(s)
            S_history.append(S)

            debug_dict["V_x"].append(V_x)
            debug_dict["V_x_next"].append(V_x_next)
            debug_dict["b1"].append(b1(t))
            debug_dict["b2"].append(b2(t))
            debug_dict["delta"].append(delta)
            debug_dict["tau"].append(tau(t))
            debug_dict["sigma"].append(sigma(t))
            debug_dict["x"].append(x)
            debug_dict["cost"].append(-reward)

        else:
            rho_history.append(rho)
            w_history.append(w)
            s_history.append(s)
            S_history.append(S)
            
        x = x_next

        # logger.info("-" *30)

def plot_debug_variables(debug_dict, s_history, S_history, config):
    warmup_period = config.get("warmup_period", 60)
    steps = np.arange(warmup_period, warmup_period + len(debug_dict["V_x"]))
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 2, 1)
    plt.plot(steps, debug_dict["V_x"], label="V_x")
    plt.plot(steps, debug_dict["V_x_next"], label="V_x_next")
    plt.legend(); plt.title("Critic Values"); plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(steps, debug_dict["delta"], label="TD Error")
    plt.title("TD Error"); plt.grid(True)

    plt.subplot(4, 2, 3)
    plt.plot(steps, debug_dict["b1"], label="b1")
    plt.plot(steps, debug_dict["b2"], label="b2")
    plt.legend(); plt.title("b1, b2"); plt.grid(True)

    plt.subplot(4, 2, 4)
    plt.plot(steps, debug_dict["tau"], label="tau")
    plt.plot(steps, debug_dict["sigma"], label="sigma")
    plt.legend(); plt.title("Hyperparameters"); plt.grid(True)

    plt.subplot(4, 2, 5)
    plt.plot(steps, debug_dict["x"], label="x")
    plt.title("Inventory state x"); plt.grid(True)
   
    plt.subplot(4, 2, 6)
    costs = np.array(debug_dict["cost"])
    avg_cost = np.cumsum(costs) / (np.arange(1, len(costs) + 1))
    plt.plot(steps, avg_cost, label="Avg Cost (cumulative)", color='green')
    plt.xlabel("Timestep")
    plt.ylabel("Cost")
    plt.title("Average Cost per Step")
    plt.legend(); plt.grid(True)

    steps = np.arange(len(s_history))    
    plt.subplot(4, 2, 7)
    plt.plot(steps, s_history, label='s', color='blue')
    plt.plot(steps, S_history, label='S', color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Policy Parameter Value")
    plt.title("Policy Parameters: s and S")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    NUM_RUNS = 5

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize accumulators
    debug_keys = ["V_x", "V_x_next", "b1", "b2", "delta", "tau", "sigma", "x", "cost"]

    debug_dict_sum = {k: None for k in debug_keys}
    s_history_sum = None
    S_history_sum = None

    for run in range(NUM_RUNS):
        logger.info(f"=== Run {run+1}/{NUM_RUNS} ===")
        env = InventoryEnv(config)

        # Initialize for each run
        rho_history, w_history = [], []
        s_history, S_history = [], []

        debug_dict = {k: [] for k in debug_keys}

        train_agent(env, config)

        # Convert to numpy arrays for accumulation
        for k in debug_keys:
            arr = np.array(debug_dict[k])
            if debug_dict_sum[k] is None:
                debug_dict_sum[k] = arr
            else:
                debug_dict_sum[k] += arr

        s_arr = np.array(s_history)
        S_arr = np.array(S_history)

        if s_history_sum is None:
            s_history_sum = s_arr
            S_history_sum = S_arr
        else:
            s_history_sum += s_arr
            S_history_sum += S_arr

    # Compute average
    debug_dict_avg = {k: debug_dict_sum[k] / NUM_RUNS for k in debug_keys}
    s_history_avg = s_history_sum / NUM_RUNS
    S_history_avg = S_history_sum / NUM_RUNS

    # Plot the averaged debug variables
    plot_debug_variables(debug_dict_avg, s_history_avg, S_history_avg, config)