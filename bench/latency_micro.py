import os
import sys
import json
import time
import socket
import argparse
import datetime
import subprocess
import statistics

import torch
import torch.distributed as dist

SIZES = [4096, 16384, 65536, 262144, 1048576]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--decode-calls", type=int, default=81)
    p.add_argument("--out", default="bench/results/gate-001.json")
    return p.parse_args()


def setup():
    if torch.cuda.is_available():
        lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr)
        dist.init_process_group("nccl")
        return torch.device("cuda", lr), torch.cuda.get_device_name(lr), "nccl"
    dist.init_process_group("gloo")
    return torch.device("cpu"), "cpu", "gloo"


def check(dev, world):
    y = torch.full((SIZES[0] // 2,), float(dist.get_rank() + 1), dtype=torch.float16, device=dev)
    dist.all_reduce(y)
    return bool((y == world * (world + 1) / 2).all().item())


def p50(nbytes, dev, iters, warmup):
    x = torch.zeros(nbytes // 2, dtype=torch.float16, device=dev)
    for _ in range(warmup):
        dist.all_reduce(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    if dev.type == "cuda":
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in ev:
            s.record()
            dist.all_reduce(x)
            e.record()
        torch.cuda.synchronize()
        t = [s.elapsed_time(e) * 1e3 for s, e in ev]
    else:
        t = []
        for _ in range(iters):
            a = time.perf_counter()
            dist.all_reduce(x)
            t.append((time.perf_counter() - a) * 1e6)
    v = torch.tensor([statistics.median(t)], dtype=torch.float64, device=dev)
    dist.all_reduce(v, op=dist.ReduceOp.MAX)
    return float(v.item())


def fit(xs, ys, n):
    xb = sum(xs) / len(xs)
    yb = sum(ys) / len(ys)
    s = sum((a - xb) * (b - yb) for a, b in zip(xs, ys)) / sum((a - xb) ** 2 for a in xs)
    a = yb - s * xb
    return a, a / (2 * (n - 1)), 2 * (n - 1) / n / s / 1e3


def label(b):
    return "{}KB".format(b // 1024) if b < 1048576 else "{}MB".format(b // 1048576)


def main():
    args = parse()
    dev, name, backend = setup()
    world = dist.get_world_size()
    ok = check(dev, world)
    runs = {b: [] for b in SIZES}
    for _ in range(args.reps):
        for b in SIZES:
            runs[b].append(p50(b, dev, args.iters, args.warmup))
    med = [statistics.median(runs[b]) for b in SIZES]
    var = max((max(runs[b]) - min(runs[b])) / statistics.median(runs[b]) * 100 for b in SIZES)
    alpha, step, link = fit(SIZES, med, world)
    dec = args.decode_calls * med[0] / 1e3
    lines = ["======{:>5} p50 {} us---------".format(label(b), " ".join("{:8.1f}".format(v) for v in runs[b])) for b in SIZES]
    lines.append("======N={} {} ok={} alpha={:.1f}us step={:.1f}us link={:.1f}GB/s decode{}={:.2f}ms var={:.2f}%---------".format(world, backend, ok, alpha, step, link, args.decode_calls, dec, var))
    if dist.get_rank() == 0:
        for l in lines:
            print(l)
        rec = {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "host": "{} ({}x {}, {})".format(socket.gethostname(), world, name, backend),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cmd": "torchrun --nproc_per_node={} {}".format(world, " ".join(sys.argv)),
            "metrics": {
                "latency_p50_variance_pct": round(var, 2),
                "output_lines": len(lines),
                "allreduce_4kb_us": round(med[0], 1),
                "alpha_us": round(alpha, 1),
                "alpha_step_us": round(step, 1),
                "link_GBps": round(link, 1),
                "decode_allreduce_ms": round(dec, 2),
                "correct": int(ok),
            },
            "runs": {label(b): [round(v, 1) for v in runs[b]] for b in SIZES},
            "doc": "bench/results/README.md",
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
