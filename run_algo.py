from typing import List
import os
import argparse
from tqdm.auto import tqdm

from env_creator import qsimpy_env_creator
from qsimpy.tasks.QTask import QTask
from qsimpy.utils.Log import Log
from schedulers import (
    BaseScheduler,
    GreedyScheduler,
    RoundRobinScheduler
)



def _summary_single_episode_result(results: dict, reward: float, episode: int) -> dict:
    waiting_time = 0.0
    execution_time = 0.0
    reschedule_count = 0
    n_qtasks_on_qnode = {}

    for i in range(5):
        n_qtasks_on_qnode["qnode_id_" + str(i)] = 0

    for res in results:
        waiting_time += res["waiting_time"]
        execution_time += res["execution_time"]
        reschedule_count += res["reschedule_count"]
        n_qtasks_on_qnode["qnode_id_" + str(res["qnode_id"])] += 1
    
    return {
        "episode": episode,
        "waiting_time": waiting_time,
        "execution_time": execution_time,
        "n_qtasks_on_qnode": n_qtasks_on_qnode,
        "reward": reward,
        "reschedule_count": reschedule_count
    }


def _save_results(results: list[dict], algorithm: str):
    import csv

    headers = ["episode", "waiting_time", "execution_time", "n_qtasks_on_qnode", "reward", "reschedule_count"]

    with open(f"results/schedulers/{algorithm}/results.csv", "w+") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


def _save_single_episode_results(results: list[dict], algorithm: str):
    import csv

    headers = ["steps", "waiting_time", "execution_time", "n_qtasks_on_qnode", "reward", "reschedule_count"]

    with open(f"results/schedulers/{algorithm}/single_episode/{results['episode']}_result.csv") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
 


def _get_scheduler(algorithm: str, **kwargs) -> BaseScheduler:
    if algorithm == "greedy":
        return GreedyScheduler(
            env=kwargs.get("env")
        )
    elif algorithm == "roundrobin":
        return RoundRobinScheduler(
            env=kwargs.get("env")
        )
    else:
        raise ValueError(
            f"Algorithm `{algorithm}` currently is not supported."
        )


def implement_scheduler(dataset: str, episode: int, algorithm: str, **kwargs):
    env = qsimpy_env_creator(
        env_config={
            "dataset": dataset,
            **kwargs
        }
    )

    scheduler = _get_scheduler(algorithm, env=env)
    results = []

    for e in (t := tqdm(range(episode), desc="Episode")):
        env.reset()
        env.setup_quantum_resources()

        e_reward = 0.0
        terminated = False

        episode_result = []

        while not terminated:
            qnode_id = scheduler.select_qnode()
            obs, reward, terminated, _, info = env.step(qnode_id)
            env.qsp_env.run()
            e_reward += reward
            scheduled_qtask: QTask = info["scheduled_qtask"]
            result = {
                "qtask_id": scheduled_qtask.id,
                "qnode_id": qnode_id,
                "arrival_time": scheduled_qtask.arrival_time,
                "execution_time": scheduled_qtask.execution_time,
                "waiting_time": scheduled_qtask.waiting_time,
                "reschedule_count": scheduled_qtask.rescheduling_count,
            }

            episode_result.append(result)
        
        t.set_description(f"Episode: {e+1} | Reward: {e_reward}")

        results.append(_summary_single_episode_result(episode_result, e_reward, e))

    env.close()

    _save_results(results, algorithm)



def main(dataset: str, episode: int, algorithms: List[str] | str, **kwargs):
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Dataset `{dataset}` doesn't exist.")
    
    if episode <= 0:
        raise ValueError(f"Episode must be larger than 0")

    if isinstance(algorithms, str):
        implement_scheduler(dataset, episode, algorithm, **kwargs)
    elif isinstance(algorithms, list):
        for algo in algorithms:
            implement_scheduler(dataset, episode, algo, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-e", "--episode", type=int, required=True)
    parser.add_argument("-a", "--algorithm", type=str, required=True)
    parser.add_argument("--obs-filter", type=str, default=None, help="Observation filter")
    parser.add_argument("--reward-filter", type=str, default=None, help="Reward filter")
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()

    dataset = args.dataset
    episode = args.episode
    algorithm = args.algorithm
    obs_filter = args.obs_filter
    reward_filter = args.reward_filter

    Log.log = args.verbose

    main(dataset, episode, algorithm, obs_filter=obs_filter, reward_filter=reward_filter)