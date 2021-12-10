from typing import KeysView

from gym.envs.registration import register

register(
    id="TrafficJunctionContinuous6-v0",
    entry_point="tjc_gym.envs:TrafficJunctionContinuousEnv",
    kwargs={"n_max": 6, "r_fov": 3},
)
