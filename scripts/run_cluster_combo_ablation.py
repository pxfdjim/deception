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
        "name": "cluster_vsa",
        "flags": [
            "--use-cluster-topk-mean-pooling",
            "--no-proto-memory-queue",
            "--use-visual-self-attn",
            "--no-instance-loss",
            "--no-topk-proto-update",
        ],
    },
    {
        "name": "cluster_topkproto",
        "flags": [
            "--use-cluster-topk-mean-pooling",
            "--no-proto-memory-queue",
            "--no-visual-self-attn",
            "--no-instance-loss",
            "--use-topk-proto-update",
        ],
    },
    {
        "name": "cluster_vsa_topkproto",
        "flags": [
            "--use-cluster-topk-mean-pooling",
            "--no-proto-memory-queue",
            "--use-visual-self-attn",
            "--no-instance-loss",
            "--use-topk-proto-update",
        ],
    },
    {
        "name": "cluster_vsa_topkproto_inst",
        "flags": [
            "--use-cluster-topk-mean-pooling",
            "--no-proto-memory-queue",
            "--use-visual-self-attn",
            "--use-instance-loss",
            "--use-topk-proto-update",
        ],
    },
]


DATASETS = {
    "dolos": "main_dolos.py",
    "seumld": "main_seumld.py",
}


def parse_gpu_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
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
    return sorted(gpus, key=lambda item: item["free"], reverse=True)


def build_command(dataset, experiment, args):
    command = [
        "python",
        DATASETS[dataset],
        "--exp-suffix",
        os.path.join("cluster_combo_ablation", experiment["name"]),
        "--cluster-topk-mean-ratio",
        str(args.cluster_topk_mean_ratio),
        "--topk-proto-ratio",
        str(args.topk_proto_ratio),
        "--topk-proto-threshold",
        str(args.topk_proto_threshold),
        "--topk-proto-warmup-epochs",
        str(args.topk_proto_warmup_epochs),
        "--proto-sep-loss-weight",
        str(args.proto_sep_loss_weight),
    ]
    command.extend(experiment["flags"])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.num_runs is not None:
        command.extend(["--num-runs", str(args.num_runs)])
    return command


def launch(dataset, experiment, gpu, args):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{dataset}_{experiment['name']}.log")
    command = build_command(dataset, experiment, args)
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
    shell_command = (
        f"cd {shlex.quote(ROOT)} && "
        f"{env_prefix} "
        + " ".join(shlex.quote(part) for part in command)
        + f" > {shlex.quote(log_path)} 2>&1"
    )
    process = subprocess.Popen(
        ["setsid", "bash", "-lc", shell_command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.pid, log_path, env_prefix + " " + " ".join(shlex.quote(part) for part in command)


def main():
    parser = argparse.ArgumentParser(description="Run cluster-centered combination ablations.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--min-free-mb", type=int, default=8000)
    parser.add_argument("--cluster-topk-mean-ratio", type=float, default=0.5)
    parser.add_argument("--topk-proto-ratio", type=float, default=0.25)
    parser.add_argument("--topk-proto-threshold", type=float, default=0.6)
    parser.add_argument("--topk-proto-warmup-epochs", type=int, default=10)
    parser.add_argument("--proto-sep-loss-weight", type=float, default=0.05)
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
