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

    def step(self, action):
        order_flag, target_inventory = action
        order_flag = 1 if order_flag >= 0.5 else 0
        target_inventory = np.clip(target_inventory, 0, self.max_inventory)

        # 주문 수행
        if order_flag:
            order_quantity = max(target_inventory - self.state, 0)
            ordering_cost = self.K + self.u * order_quantity
        else:
            order_quantity = 0
            ordering_cost = 0

        if self.regime_switching:
            regime_index = (self.t // self.switching_period) % len(self.regime_sequence)
            self.alpha, self.beta = self.regime_sequence[regime_index]

        demand = np.random.gamma(self.alpha, 1.0 / self.beta)
        self.t += 1
        next_state = self.state + order_quantity - demand

        # 비용 계산
        holding_cost = self.h * max(next_state, 0)
        backlog_cost = self.b * max(-next_state, 0)
        total_cost = ordering_cost + holding_cost + backlog_cost

        reward = -total_cost  # 보상은 비용의 음수
        self.state = next_state

        return np.array([self.state], dtype=np.float32), reward, False, {}

    def reset(self):
        self.state = 0.0  # 초기 재고는 0
        return np.array([self.state], dtype=np.float32)

    def render(self, mode='human'):
        print(f"재고 수준: {self.state:.2f}")


