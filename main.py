import numpy as np
from loguru import logger
import yaml
from inventory_env import InventoryEnv
import matplotlib.pyplot as plt

def phi(x):
    x = x/5
    return np.array([x, x**2, x**3, x**4])

def sigmoid(x, tau):
    z = np.clip(x / tau, -100, 100)  # prevent overflow in exp
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x, tau):
    s = sigmoid(x, tau)
    return (1 / tau) * s * (1 - s)

def estimate_gamma_params(samples):
    if len(samples) < 2:
        return 2.0, 1.0
    mean = np.mean(samples)
    var = np.var(samples)
    alpha_hat = mean**2 / var if var > 0 else 2.0
    beta_hat = mean / var if var > 0 else 1.0
    return alpha_hat, beta_hat

def update_value_function(w, rho, x, x_next, reward, gamma1, gamma2, t):
    """
    Update the value function weights and average reward using TD error.

    Args:
        w: critic weights
        rho: average reward estimate
        x, x_next: current and next state
        reward: observed reward
        phi: feature mapping function
        gamma1, gamma2: learning rate schedules
        t: current timestep

    Returns:
        updated w, rho, delta
    """
    V_x = w @ phi(x)
    V_x_next = w @ phi(x_next)
    delta = reward - rho + V_x_next - V_x

    delta = np.clip(delta, -10.0, 10.0)

    w += gamma1(t) * delta * phi(x)
    rho += gamma2(t) * (reward + V_x_next - V_x - rho)

    # logger.info(
    #     f"[{t}] x={x:.4f}, V_x={V_x:.4f}, V_x_next={V_x_next:.4f}, "
    #     f"delta={delta:.4f}, "
    #     f"w={np.array2string(w, precision=4)}, "
    #     f"rho={rho:.4f}"
    # )

    return w, rho, V_x, V_x_next, delta

def update_policy_parameters(s, S, x, x_next, w, b1, b2, t, alpha_hat, beta_hat, tau, S_tilde, max_inventory):

    # Sampling Bernoulli variables (η_S and η_s)
    eta_S = np.random.binomial(1, 0.5)
    eta_s = np.random.binomial(1, 0.5)

    # Sampling z_s and z_S based on transition model
    # z_S는 S를 조금 바꾸었다면 다음 상태 x'가 어떻게 나왔을지 추정하는 가상의 x' (x'는 실제 trajectory에서 관측된 상태)
    # dP(x'|S_tilde)/dS를 직접 구하기 어려우므로 샘플 기반으로 근사
    if eta_S == 0:
        z_S = S_tilde - np.random.gamma(alpha_hat-1, 1 / beta_hat)
    else:
        z_S = S_tilde - np.random.gamma(alpha_hat, 1 / beta_hat)

    if eta_s == 0:
        z_s = x - np.random.gamma(alpha_hat-1, 1 / beta_hat)
    else:
        z_s = S_tilde - np.random.gamma(alpha_hat, 1 / beta_hat)

    # s를 조금 바꾸었을 때 그것이 x → x'로 갈 확률을 얼마나 바꾸는지 계산
    logger.info(f"[{t}] z_s: {z_s:.4f}, z_S: {z_S:.4f}, eta_s: {eta_s}, eta_S: {eta_S}")

    # Compute policy function f(x, s) and its derivative
    f_xs = sigmoid(x - s, tau)
    df_dy = sigmoid_derivative(x - s, tau)

    # Get value estimates from critic
    V_zs = w @ phi(z_s)
    V_zS = w @ phi(z_S)

    # Apply policy update rules as per SRL-FSA
    S -= b1(t) * beta_hat * (1 - f_xs) * ((-1) ** eta_S) * V_zS
    s -= b2(t) * df_dy * ((-1) ** eta_s) * V_zs

    S = np.clip(S, 0, max_inventory)  # Ensure S <= max_inventory
    s = np.clip(s, 0, S)  # Ensure s <= S

    return s, S, V_zs, V_zS

def train_agent(env, config):
    
    # config에서 초기 파라미터 불러오기
    max_inventory = config.get("max_inventory", 50)
    s = config["s_init"]
    S = config["S_init"]
    rho = config["rho_init"]
    sigma = config["sigma"]
    tau = config["tau"]
    sigma_decay = config["sigma_decay"]
    tau_decay = config["tau_decay"]

    b1 = lambda t: config["b1"]
    b2 = lambda t: config["b2"]
    gamma1 = lambda t: config["gamma1_base"] / (1 + config["gamma_decay"] * t)
    gamma2 = lambda t: config["gamma2_base"] / (1 + config["gamma_decay"] * t)
    
    warmup_period = config.get("warmup_period", 60)
    train_period = config.get("train_period", 1000)
    episodes = len(config["regime_sequence"]) * (warmup_period + train_period)

    w = np.zeros(4)
    demand_history = []

    obs = env.reset()
    x = obs[0]

    for t in range(100):
        # step 2 : observe the transitioned state and corresponding reward after taking action at given state x_t
        # 정책 기반 행동 선택
        # 현재 재고 x가 reorder point s보다 작은지 soft하게 판별
        # deterministic (𝑠, 𝑆) 정책은 MDP에서 단일 커뮤니케이션 클래스를 만들지 않기 때문에 → 일반적인 RL convergence 조건을 만족하지 않음
        # 그래서 noise와 soft decision을 추가해서 → 모든 상태 reachable하게 만듦
        prob = sigmoid(x - s, tau)

        if np.random.rand() > prob: # 확률적으로 주문 여부 결정
            noise = np.random.normal(0, sigma)
            S_tilde = S + noise
            a = max(S_tilde - x, 0) # 약간의 noise를 추가한 S tilde까지 주문
        else:
            a = 0 # 주문 안함

        # 환경 상호작용
        d, x_next, reward = env.step(a)

        # step 3 : attain realized demand and adaptively estimate the distributional parameters
        demand_history.append(max(d, 0))  # 수요는 비음수
        alpha_hat, beta_hat = estimate_gamma_params(demand_history)
        logger.info(f"[{t}] Demand: {d:.4f}, Estimated Params: alpha={alpha_hat:.4f}, beta={beta_hat:.4f}")

        # step 4 : update the relative value function
        w, rho, V_x, V_x_next, delta = update_value_function(w, rho, x, x_next, reward, gamma1, gamma2, t)
        
        rho_history.append(rho)
        w_history.append(w)

        # step 5 : update the policy parameters
        s, S, V_zs, V_zS = update_policy_parameters(s, S, x, x_next, w, b1, b2, t, alpha_hat, beta_hat, tau, S_tilde, max_inventory)

        logger.info(
            f"[{t}] "
            f"V_x={V_x:.4f}, V_x_next={V_x_next:.4f}, "
            f"rho={rho:.4f}, "
            f"s={s:.4f}, S={S:.4f}, "
            f"V_zs={V_zs:.4f}, V_zS={V_zS:.4f}, "
            f"delta={delta:.4f}, "
            f"alpha_hat={alpha_hat:.4f}, beta_hat={beta_hat:.4f}, "
            f"tau={tau:.4f}, sigma={sigma:.4f}"
        )

        s_history.append(s)
        S_history.append(S)

        # step 6 : update the hyperparameters
        sigma *= sigma_decay
        tau *= tau_decay

        debug_dict["V_x"].append(V_x)
        debug_dict["V_x_next"].append(V_x_next)
        debug_dict["V_zs"].append(V_zs)
        debug_dict["V_zS"].append(V_zS)
        debug_dict["delta"].append(delta)
        debug_dict["tau"].append(tau)
        debug_dict["sigma"].append(sigma)
        debug_dict["x"].append(x)

        x = x_next
        

def plot_training_history(s_history, S_history, rho_history):
    steps = np.arange(len(s_history))

    plt.figure(figsize=(12, 5))

    # (1) s, S plot
    plt.subplot(1, 2, 1)
    plt.plot(steps, s_history, label='s', color='blue')
    plt.plot(steps, S_history, label='S', color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Policy Parameter Value")
    plt.title("Policy Parameters: s and S")
    plt.legend()
    plt.grid(True)

    # (2) rho (average reward estimate)
    plt.subplot(1, 2, 2)
    plt.plot(steps, rho_history, color='green')
    plt.xlabel("Timestep")
    plt.ylabel("rho (Avg. Reward)")
    plt.title("Estimated Average Reward (rho)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_debug_variables(debug_dict):
    steps = np.arange(len(debug_dict["V_x"]))
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 2, 1)
    plt.plot(steps, debug_dict["V_x"], label="V_x")
    plt.plot(steps, debug_dict["V_x_next"], label="V_x_next")
    plt.legend(); plt.title("Critic Values"); plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(steps, debug_dict["delta"], label="TD Error")
    plt.title("TD Error"); plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(steps, debug_dict["V_zs"], label="V_zs")
    plt.plot(steps, debug_dict["V_zS"], label="V_zS")
    plt.legend(); plt.title("V(z_s), V(z_S)"); plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(steps, debug_dict["tau"], label="tau")
    plt.plot(steps, debug_dict["sigma"], label="sigma")
    plt.legend(); plt.title("Hyperparameters"); plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(steps, debug_dict["x"], label="x")
    plt.title("Inventory state x"); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    rho_history, w_history = [], []
    s_history, S_history = [], []

    debug_dict = {
        "V_x": [], "V_x_next": [], "V_zs": [], "V_zS": [],
        "delta": [], "tau": [], "sigma": [], "x": []
    }

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = InventoryEnv(config)
    train_agent(env, config)

    # plot_debug_variables(debug_dict)
    plot_training_history(s_history, S_history, rho_history)