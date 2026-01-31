from rejax.algos import DQN, IQN, PQN, SAC, TD3, Algorithm
from rejax.algos.ppo import PPO


_algos = {
    "dqn": DQN,
    "iqn": IQN,
    "ppo": PPO,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "DQN",
    "IQN",
    "PPO",
    "PQN",
    "SAC",
    "TD3",
    "get_algo",
]
