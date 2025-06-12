import numpy as np
import pandas as pd
from loguru import logger
import yaml
from inventory_env import InventoryEnv
import matplotlib.pyplot as plt
from main_nonstationary import *

plt.rcParams['figure.figsize'] = (12, 8)


if __name__ == "__main__":
    config_list = ["config1.yaml","config2.yaml","config3.yaml"]
    for ex in config_list :
        with open(ex, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        env = InventoryEnv(config)
        debug_dict = train_agent_base(env, config)
        plot_debug_variables(debug_dict)

    