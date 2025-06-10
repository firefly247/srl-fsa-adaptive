import numpy as np
import gym
from gym import spaces

class InventoryEnv(gym.Env):

    def __init__(self, config):
        super(InventoryEnv, self).__init__()

        # config 딕셔너리에서 하이퍼파라미터를 받아옴
        self.alpha = config.get("alpha", 2.0)
        self.beta = config.get("beta", 1.0)
        self.h = config.get("holding_cost", 0.1)
        self.b = config.get("backlog_cost", 0.5)
        self.K = config.get("setup_cost", 0.1)
        self.u = config.get("unit_cost", 0.3)
        self.max_inventory = config.get("max_inventory", 50)

        self.regime_switching = config.get("regime_switching", False)
        self.switching_period = config.get("switching_period", 1000)
        self.regime_sequence = config.get("regime_sequence", [[2.0, 1.0]])
        self.t = 0  # time step

        # 상태 공간: 재고 수준 하나 (연속형)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # 행동 공간: [주문 여부 flag (0~1), 목표재고수준 S]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]),
                                       high=np.array([1.0, float(self.max_inventory)]), dtype=np.float32)

        self.reset()

    def truncated_poisson_sample(self, lam, max_val=7):
        while True:
            sample = np.random.poisson(lam)
            if sample <= max_val:
                return sample
        
    def step(self, action):
        # 주문 수행
        if action > 0:
            ordering_cost = self.K + self.u * action
        else:
            ordering_cost = 0

        if self.regime_switching:
            regime_index = (self.t // self.switching_period) % len(self.regime_sequence)
            self.alpha, self.beta = self.regime_sequence[regime_index]

        # demand = np.random.gamma(self.alpha, 1.0 / self.beta)
        demand = self.truncated_poisson_sample(self.alpha)  # 포아송 분포로 수요 생성
        self.t += 1
        next_state = self.state + action - demand

        # 비용 계산
        holding_cost = self.h * max(next_state, 0)
        backlog_cost = self.b * max(-next_state, 0)
        total_cost = ordering_cost + holding_cost + backlog_cost

        reward = -total_cost  # 보상은 비용의 음수
        self.state = next_state

        return demand, self.state, reward

    def reset(self):
        self.state = 0.0  # 초기 재고는 0
        return np.array([self.state], dtype=np.float32)

    def render(self, mode='human'):
        print(f"재고 수준: {self.state:.2f}")


