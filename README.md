# tjc-gym

Traffic Juntion Continuous (tjc) is an extended version of the environment Traffic Junction, defined in [Learning Multiagent Communication
with Backpropagation](https://arxiv.org/pdf/1605.07736.pdf). The environment is based on OpenAI Gym and has an continouos action space where the original is discrete.

# Installation

Through PyPi:

```bash
pip install tjc-gym
```

Or directly by cloning repo:

```bash
git clone https://github.com/jakobdybdahl/tjc-gym.git
cd tjc-gym
pip install -e .
```

# Usage

```python
import gym

env = gym.make('tjc_gym:TrafficJunction6-v0')
done = [False] * env.n_agents
score = 0

obs = env.reset()
while not all(done):
  env.render()
  actions = [acsp.sample() for ascp in env.action_space]
  obs_, rewards, done, info = env.step(actions)
  score += sum(rewards)


```

# Reference

**TO DO**

# Acknowledgement

This environment was developed by [RasmusThorsen](https://github.com/RasmusThorsen) and [jakobdybdahl](https://github.com/jakobdybdahl) to complement our master thesis at @ [Aarhus University](https://www.au.dk/).
