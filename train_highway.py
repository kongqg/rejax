import timeit

import jax
import jax.numpy as jnp
import yaml
from matplotlib import pyplot as plt

from rejax.get_highway import get_algo
import wandb
import time


def main(algo_str, config, seed_id, num_seeds, time_fit):
    eval_taus = jnp.array([0.6, 0.7, 0.8, 0.9])
    eval_seeds_count = num_seeds
    num_taus = len(eval_taus)

    algo_cls = get_algo(algo_str)
    algo = algo_cls.create(**config)

    # 2. 定义向量化训练逻辑
    def train_with_tau(tau, keys):
        curr_algo = algo.replace(tau=tau)
        return jax.vmap(algo_cls.train, in_axes=(None, 0))(curr_algo, keys)

    final_vmap_train = jax.jit(jax.vmap(train_with_tau, in_axes=(0, 0)))

    # 3. 准备随机数种子
    key = jax.random.PRNGKey(seed_id)
    subkeys = jax.random.split(key, num_taus * eval_seeds_count)
    subkeys = subkeys.reshape(num_taus, eval_seeds_count, -1)


    print(f"Starting training for {num_taus} taus with {eval_seeds_count} seeds each...")
    start_time = time.time()
    ts, (all_lengths, all_returns) = final_vmap_train(eval_taus, subkeys)
    all_returns.block_until_ready()
    end_time = time.time()

    duration = end_time - start_time
    total_seeds = num_seeds

    print("-" * 30)
    print(f"训练总耗时: {duration:.2f} 秒")
    print(f"总 Seed 数量 (num_taus * num_seeds): {total_seeds}")
    print("-" * 30)


    print("Training finished.")


    import numpy as np
    returns_np = np.array(all_returns)  # 形状: (num_taus, num_seeds, num_steps, num_envs)

    for i, tau_val in enumerate(eval_taus):
        tau_float = float(tau_val)

        # 为每个 tau 值开启一个独立的 Run
        with wandb.init(
                project="rejax-kongqg",
                config={**config, "tau": tau_float, "algo_type": "highway"},
                name=f"{config['env']}-tau-{tau_float:.1f}",
                group=config['env'],
                reinit=True
        ):
            avg_returns_history = returns_np[i].mean(axis=(0, 2))

            print(f"Logging data for tau={tau_float:.1f}...")
            for step_idx, val in enumerate(avg_returns_history):
                current_step = step_idx * algo.eval_freq
                wandb.log({"eval/mean_returns": float(val)}, step=int(current_step))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gymnax/cartpole.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--time-fit",
        action="store_true",
        help="Time how long it takes to fit the agent by fitting 3 times.",
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to roll out.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())[args.algorithm]


    main(
        args.algorithm,
        config,
        args.seed_id,
        args.num_seeds,
        args.time_fit,
    )
