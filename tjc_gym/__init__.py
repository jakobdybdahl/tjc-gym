from typing import KeysView

from gym.envs.registration import register

register(
    id="TrafficJunctionContinuous6-v0",
    entry_point="tjc_gym.envs:TrafficJunctionContinuousEnv",
    kwargs={"n_max": 6, "r_fov": 3},
)
register(
    id="TrafficJunctionContinuous1-v0",
    entry_point="tjc_gym.envs:TrafficJunctionContinuousEnv",
    kwargs={"n_max": 1, "r_fov": 3},
)
register(
    id="TrafficJunctionContinuous2-v0",
    entry_point="tjc_gym.envs:TrafficJunctionContinuousEnv",
    kwargs={"n_max": 2, "r_fov": 3},
)
register(
    id="TrafficJunctionContinuous4-v0",
    entry_point="tjc_gym.envs:TrafficJunctionContinuousEnv",
    kwargs={"n_max": 4, "r_fov": 3},
)
