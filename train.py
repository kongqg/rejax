import timeit

import jax
import jax.numpy as jnp
import yaml
from matplotlib import pyplot as plt

from rejax import get_algo
from src.rejax.evaluate import evaluate_grid
import wandb


def main(algo_str, config, seed_id, num_seeds, time_fit):
    wandb.init(
        project="rejax-kongqg",
        config=config,
        name=f"{config.ppo.env}_{config.ppo.tau}",
        group=config.ppo.env,  #
    )
    algo_cls = get_algo(algo_str)
    algo = algo_cls.create(**config)
    print(algo.config)

    old_eval_callback = algo.eval_callback
    eval_taus = jnp.array([0.6, 0.7, 0.8, 0.9])
    eval_seeds = jnp.array([111, 222, 333, 444, 555, 666, 777, 888])
    def eval_callback(algo, ts, rng):
        act = algo.make_act(ts)
        lengths, returns = evaluate_grid(
            act,
            rng,
            algo.env,
            algo.env_params,
            eval_taus,
            eval_seeds,
            algo.env_params.max_steps_in_episode
        )
        # 按照tau的值 取mean
        mean_returns_per_tau = returns.mean(axis=1)
        jax.debug.print(
            "Step {}: Mean returns for taus {} are {}",
            ts.global_step,
            eval_taus,
            mean_returns_per_tau
        )
        if wandb.run is not None:
            metrics = {"global_step": ts.global_step}
            for i, tau in enumerate(eval_taus):
                metrics[f"eval/return_tau_{tau:.2f}"] = mean_returns_per_tau[i]
            wandb.log(metrics)
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

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
