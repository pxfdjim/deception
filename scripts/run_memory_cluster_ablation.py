#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import time


ROOT = "/home/pengxf/work/TDD/Video_MAEV2/Deception"
LOG_DIR = os.path.join(ROOT, "experiments", "component_ablation_logs")


EXPERIMENTS = [
    {
        "name": "memq",
        "flags": ["--use-proto-memory-queue", "--no-cluster-topk-mean-pooling"],
    },
    {
        "name": "cluster_topk",
        "flags": ["--no-proto-memory-queue", "--use-cluster-topk-mean-pooling"],
    },
    {
        "name": "memq_cluster_topk",
        "flags": ["--use-proto-memory-queue", "--use-cluster-topk-mean-pooling"],
    },
]


DATASETS = {
    "dolos": {
        "entry": "main_dolos.py",
    },
    "seumld": {
        "entry": "main_seumld.py",
    },
}


def parse_gpu_snapshot():
    query = "index,memory.used,memory.free,utilization.gpu"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in output.strip().splitlines():
        index, used, free, util = [item.strip() for item in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "used": int(used),
                "free": int(free),
                "util": int(util),
            }
        )
    return sorted(gpus, key=lambda item: (item["free"], -item["used"]), reverse=True)


def build_command(dataset, experiment, gpu, args):
    dataset_cfg = DATASETS[dataset]
    command = [
        "python",
        dataset_cfg["entry"],
        "--no-visual-self-attn",
        "--no-instance-loss",
        "--no-topk-proto-update",
        "--exp-suffix",
        os.path.join("memory_cluster_ablation", experiment["name"]),
        "--proto-memory-queue-size",
        str(args.proto_memory_queue_size),
        "--proto-memory-momentum",
        str(args.proto_memory_momentum),
        "--cluster-topk-mean-ratio",
        str(args.cluster_topk_mean_ratio),
    ]
    command.extend(experiment["flags"])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.num_runs is not None:
        command.extend(["--num-runs", str(args.num_runs)])

    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
    return f"{env_prefix} " + " ".join(shlex.quote(part) for part in command)


def launch(dataset, experiment, gpu, args):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{dataset}_{experiment['name']}.log")
    command = build_command(dataset, experiment, gpu, args)
    full_command = f"cd {shlex.quote(ROOT)} && {command} > {shlex.quote(log_path)} 2>&1"
    process = subprocess.Popen(
        ["setsid", "bash", "-lc", full_command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.pid, log_path, command


def main():
    parser = argparse.ArgumentParser(description="Run memory queue and cluster top-k ablations.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--min-free-mb", type=int, default=6000)
    parser.add_argument("--proto-memory-queue-size", type=int, default=512)
    parser.add_argument("--proto-memory-momentum", type=float, default=0.5)
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    args = parser.parse_args()

    jobs = [(dataset, experiment) for dataset in args.datasets for experiment in EXPERIMENTS]
    gpu_snapshot = parse_gpu_snapshot()
    free_gpus = [gpu for gpu in gpu_snapshot if gpu["free"] >= args.min_free_mb]
    if not free_gpus:
        free_gpus = gpu_snapshot[:1]

    launched = []
    for idx, (dataset, experiment) in enumerate(jobs):
        gpu = free_gpus[idx % len(free_gpus)]["index"]
        pid, log_path, command = launch(dataset, experiment, gpu, args)
        launched.append((dataset, experiment["name"], gpu, pid, log_path, command))
        time.sleep(2)

    print("Launched experiments:")
    for dataset, name, gpu, pid, log_path, command in launched:
        print(f"- {dataset}/{name}: gpu={gpu}, pid={pid}, log={log_path}")
        print(f"  {command}")


if __name__ == "__main__":
    main()
