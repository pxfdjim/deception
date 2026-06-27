#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "none",
        "use_visual_self_attn": False,
        "use_instance_loss": False,
    },
    {
        "name": "visual_self_attn",
        "use_visual_self_attn": True,
        "use_instance_loss": False,
    },
    {
        "name": "instance_loss",
        "use_visual_self_attn": False,
        "use_instance_loss": True,
    },
    {
        "name": "both",
        "use_visual_self_attn": True,
        "use_instance_loss": True,
    },
]

DATASET_SCRIPTS = {
    "dolos": "main_dolos.py",
    "seumld": "main_seumld.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run component ablations on the freest GPUs.")
    parser.add_argument("--datasets", default="dolos,seumld", help="Comma-separated subset: dolos,seumld")
    parser.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-free-mb", type=int, default=12000)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--launch-delay", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def query_gpu_free_memory():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    gpus = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index, free_mb = [part.strip() for part in line.split(",")]
        gpus.append((int(index), int(free_mb)))
    return sorted(gpus, key=lambda item: item[1], reverse=True)


def build_command(args, dataset, experiment):
    command = [
        args.python,
        DATASET_SCRIPTS[dataset],
        "--exp-suffix",
        f"component_ablation/{experiment['name']}",
    ]
    if experiment["use_visual_self_attn"]:
        command.append("--use-visual-self-attn")
    else:
        command.append("--no-visual-self-attn")

    if experiment["use_instance_loss"]:
        command.append("--use-instance-loss")
    else:
        command.append("--no-instance-loss")

    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.num_runs is not None:
        command.extend(["--num-runs", str(args.num_runs)])
    return command


def build_jobs(args):
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DATASET_SCRIPTS))
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")

    jobs = []
    for experiment in EXPERIMENTS:
        for dataset in datasets:
            jobs.append(
                {
                    "dataset": dataset,
                    "experiment": experiment["name"],
                    "command": build_command(args, dataset, experiment),
                }
            )
    return jobs


def launch_job(args, job, gpu_index, log_dir):
    log_path = log_dir / f"{job['dataset']}_{job['experiment']}.log"
    log_file = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    print(f"[launch] gpu={gpu_index} log={log_path} cmd={' '.join(job['command'])}", flush=True)
    process = subprocess.Popen(
        job["command"],
        cwd=args.workdir,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "process": process,
        "gpu": gpu_index,
        "job": job,
        "log_file": log_file,
        "log_path": log_path,
    }


def main():
    args = parse_args()
    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir does not exist: {workdir}")
    args.workdir = str(workdir)

    jobs = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] {' '.join(job['command'])}")
        return

    log_dir = workdir / "experiments" / "component_ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(jobs)
    running = []
    print(f"Queued {len(pending)} jobs. One job will be assigned per GPU.", flush=True)

    while pending or running:
        still_running = []
        for item in running:
            return_code = item["process"].poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["log_file"].close()
            job = item["job"]
            print(
                f"[done] gpu={item['gpu']} dataset={job['dataset']} "
                f"experiment={job['experiment']} return_code={return_code} log={item['log_path']}",
                flush=True,
            )
            if return_code != 0:
                print(f"[warn] Job failed: {job['dataset']} {job['experiment']}", flush=True)
        running = still_running

        running_gpus = {item["gpu"] for item in running}
        available_gpus = [
            (gpu, free_mb)
            for gpu, free_mb in query_gpu_free_memory()
            if gpu not in running_gpus and free_mb >= args.min_free_mb
        ]

        while pending and available_gpus:
            gpu_index, free_mb = available_gpus.pop(0)
            job = pending.popleft()
            print(f"[select] gpu={gpu_index} free_mb={free_mb}", flush=True)
            running.append(launch_job(args, job, gpu_index, log_dir))
            time.sleep(args.launch_delay)

        if pending:
            print(
                f"[wait] pending={len(pending)} running={len(running)} "
                f"min_free_mb={args.min_free_mb}",
                flush=True,
            )
        if pending or running:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
