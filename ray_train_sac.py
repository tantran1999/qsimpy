import os

import ray
from ray.rllib.algorithms import SACConfig
from ray import train, tune
from ray.tune import register_env

from env_creator import qsimpy_env_creator


os.environ["RAY_DEDUP_LOGS"] = "0"


def main(framework: str, stop_timesteps: int, stop_iters: int):
    register_env("QSimPyEnv", qsimpy_env_creator)

    config = (
        SACConfig()
        .environment(
            env="QSimPyEnv",
            env_config={
                "dataset": "/Users/tantran5/Personal/quantumai-cloud/aiqc/qsimpy/qdataset/qsimpyds_1000_sub_24.csv",
                "obs_filter": "rescale_-1_1",
                "reward_filter": None
            }
        )
        .training(
            gamma=0.8,
            lr=tune.grid_search([0.01]),
            train_batch_size=tune.grid_search([72]),
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 60000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.5,
                "prioritized_replay_eps": 3e-6,
            },
            policy_model_config={
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 256],
            },
            q_model_config={
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 256],
            }
        )
        .framework(framework)
        .resources(num_cpus_per_worker=2)
    )

    result_directory = os.path.join(os.getcwd(), "results")

    try:
        tuner = tune.Tuner(
            "SAC",
            param_space=config.to_dict(),
            run_config=train.RunConfig(
                name="SAC_QCE_1000",
                storage_path=f"file://{result_directory}",
                checkpoint_config=train.CheckpointConfig(num_to_keep=10),
                stop={
                    "timesteps_total": stop_timesteps,
                    "training_iteration": stop_iters,
                }
            )
        )
        results = tuner.fit()
    except Exception as e:
        raise e

    try:
        # Get the best result based on a particular metric
        best_result = results.get_best_result(
            metric="env_runners/episode_return_mean", mode="max"
        )
        best_checkpoint = best_result.checkpoint
        best_checkpoint.to_directory("results/checkpoints")
    except Exception as e:
        raise e

    ray.shutdown()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=100, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
    )
    args = parser.parse_args()

    main(
        framework=args.framework,
        stop_timesteps=args.stop_timesteps,
        stop_iters=args.stop_iters
    )