import timeit

import jax
import jax.numpy as jnp
import yaml
from matplotlib import pyplot as plt

from rejax.get_ppo import get_algo
import wandb


def main(algo_str, config, seed_id, num_seeds, time_fit):
    wandb.init(
        project="rejax-kongqg",
        config=config,
        name=f"{config['env']}",
        group=config['env'],  #
    )
    algo_cls = get_algo(algo_str)
    algo = algo_cls.create(**config)
    print(algo.config)

    old_eval_callback = algo.eval_callback
    eval_seeds = jnp.array([111, 222, 333, 444, 555, 666, 777, 888])
    train_num_seeds = len(eval_seeds)

    def eval_callback(algo, ts, rng):
        # 1. (128,)
        lengths, returns = old_eval_callback(algo, ts, rng)

        # 2.
        avg_lengths = lengths.mean()
        avg_returns = returns.mean()

        # 3.
        def log_to_wandb(step, l_arr, r_arr):
            import numpy as np
            mean_step_length = np.mean(l_arr)
            mean_return = np.mean(r_arr)

            # 记录到 wandb
            wandb.log({
                "eval/mean_returns": float(mean_return),
                "eval/mean_episode_lengths": float(mean_step_length)
            }, step=int(step[0]))  # 使用第一个 seed 的 step 作为步数

        # 4. 使用 io_callback 触发日志记录
        jax.experimental.io_callback(
            log_to_wandb,
            None,
            ts.global_step,  # 传入当前训练步数
            avg_lengths,  # 传入 (num_seeds,) 形状的长度数组
            avg_returns  # 传入 (num_seeds,) 形状的奖励数组
        )

        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, train_num_seeds)

    vmap_train = jax.jit(jax.vmap(algo_cls.train, in_axes=(None, 0)))
    ts, (_, returns) = vmap_train(algo, keys)
    returns.block_until_ready()

    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * algo.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(algo, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(
            f"{time:.1f} seconds total, equalling to "
            f"{time / num_seeds:.1f} seconds per seed"
        )

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())


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