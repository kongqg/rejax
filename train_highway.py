import timeit

import jax
import jax.numpy as jnp
import yaml
from matplotlib import pyplot as plt

from rejax.algos.get_highway import get_algo
import wandb


def main(algo_str, config, seed_id, num_seeds, time_fit):
    wandb.init(
        project="rejax-kongqg",
        config=config,
        name = f"{config['env']}",
        group=config['env'],  #
    )
    eval_taus = jnp.array([0.6, 0.7, 0.8, 0.9])
    eval_seeds = jnp.array([111, 222, 333, 444, 555, 666, 777, 888])
    num_taus = len(eval_taus)
    num_seeds_per_tau = len(eval_seeds)

    algo_cls = get_algo(algo_str)
    base_config = config.copy()
    algo = algo_cls.create(**base_config)
    original_eval_callback = algo.eval_callback

    def wandb_avg_callback(algo, ts, rng):
        lengths, returns = original_eval_callback(algo, ts, rng)
        avg_lengths = lengths.mean()
        avg_returns = returns.mean()

        def log_to_wandb(step, tau_now, lengths_arr, returns_arr):
            import numpy as np
            lengths_arr = np.array(lengths_arr)
            returns_arr = np.array(returns_arr)
            step_arr = np.array(step)
            tau_now = float(np.array(tau_now))  # 兼容 DeviceArray

            try:
                l_mat = lengths_arr.reshape(num_taus, num_seeds_per_tau)
                r_mat = returns_arr.reshape(num_taus, num_seeds_per_tau)
                s_mat = step_arr.reshape(num_taus, num_seeds_per_tau)

                mean_lengths = l_mat.mean(axis=1)
                mean_returns = r_mat.mean(axis=1)
                global_step = int(s_mat[0, 0])
            except ValueError:
                mean_lengths = lengths_arr
                mean_returns = returns_arr
                global_step = int(step_arr.flat[0])

            log_dict = {}

            if np.ndim(mean_returns) == 0:
                suffix = f"tau_{tau_now:.1f}"
                log_dict[f"return/{suffix}"] = float(mean_returns)
                log_dict[f"length/{suffix}"] = float(mean_lengths)
            elif mean_returns.shape[0] == len(eval_taus):
                for i, tau_val in enumerate(eval_taus):
                    suffix = f"tau_{tau_val:.1f}"
                    log_dict[f"return/{suffix}"] = float(mean_returns[i])
                    log_dict[f"length/{suffix}"] = float(mean_lengths[i])
            else:
                # 兜底：不符合预期形状就记录一个平均
                log_dict["return/avg"] = float(np.mean(mean_returns))
                log_dict["length/avg"] = float(np.mean(mean_lengths))

            wandb.log(log_dict, step=global_step)

        jax.experimental.io_callback(
            log_to_wandb,
            None,
            ts.global_step,
            algo.tau,
            avg_lengths,
            avg_returns
        )
        return lengths, returns

    algo = algo.replace(eval_callback=wandb_avg_callback)
    print(algo.config)

    def train_with_tau(tau, keys):
        curr_algo = algo.replace(tau=tau)
        return jax.vmap(algo_cls.train, in_axes=(None, 0))(curr_algo, keys)
    final_vmap_train = jax.jit(jax.vmap(train_with_tau, in_axes=(0, 0)))

    key = jax.random.PRNGKey(seed_id)
    subkeys = jax.random.split(key, num_taus * num_seeds_per_tau)
    subkeys = subkeys.reshape(num_taus, num_seeds_per_tau, -1)

    print(f"Starting training for {num_taus} taus with {num_seeds_per_tau} seeds each...")
    ts, (all_lengths, all_returns) = final_vmap_train(eval_taus, subkeys)
    all_returns.block_until_ready()

    print("Training finished.")
    wandb.finish()


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
