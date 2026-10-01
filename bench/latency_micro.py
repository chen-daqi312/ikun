import os
import sys
import json
import time
import shutil
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
    p.add_argument("--chunks", type=int, default=10)
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


def timed(x, n, dev):
    if dev.type == "cuda":
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        for s, e in ev:
            s.record()
            dist.all_reduce(x)
            e.record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in ev]
    out = []
    for _ in range(n):
        a = time.perf_counter()
        dist.all_reduce(x)
        out.append((time.perf_counter() - a) * 1e6)
    return out


def sweep(dev, reps, iters, chunks, warmup):
    buf = {b: torch.zeros(b // 2, dtype=torch.float16, device=dev) for b in SIZES}
    for b in SIZES:
        timed(buf[b], warmup, dev)
    dist.barrier()
    s = {(b, r): [] for b in SIZES for r in range(reps)}
    for _ in range(chunks):
        for r in range(reps):
            for b in SIZES:
                s[(b, r)] += timed(buf[b], max(1, iters // chunks), dev)
    return {k: statistics.median(v) for k, v in s.items()}


def reduce_max(vals, dev):
    t = torch.tensor(vals, dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def fit(xs, ys, n):
    xb = sum(xs) / len(xs)
    yb = sum(ys) / len(ys)
    s = sum((a - xb) * (b - yb) for a, b in zip(xs, ys)) / sum((a - xb) ** 2 for a in xs)
    return yb - s * xb, (yb - s * xb) / (2 * (n - 1)), 2 * (n - 1) / n / s / 1e3


def label(b):
    return "{}KB".format(b // 1024) if b < 1048576 else "{}MB".format(b // 1048576)


def driver():
    exe = shutil.which("ixsmi") or shutil.which("nvidia-smi")
    if exe is None:
        return "none"
    r = subprocess.run([exe], capture_output=True, text=True)
    hit = [l for l in r.stdout.splitlines() if "Driver Version" in l]
    return hit[0].strip() if hit else "unknown"


def main():
    args = parse()
    dev, name, backend = setup()
    world = dist.get_world_size()
    ok = check(dev, world)
    med = sweep(dev, args.reps, args.iters, args.chunks, args.warmup)
    flat = reduce_max([med[(b, r)] for b in SIZES for r in range(args.reps)], dev)
    per = {b: flat[i * args.reps:(i + 1) * args.reps] for i, b in enumerate(SIZES)}
    mid = [statistics.median(per[b]) for b in SIZES]
    var = max((max(per[b]) - min(per[b])) / statistics.median(per[b]) * 100 for b in SIZES)
    alpha, step, link = fit(SIZES, mid, world)
    dec = args.decode_calls * mid[0] / 1e3
    out = ["======{:>5} p50 {} us---------".format(label(b), " ".join("{:8.1f}".format(v) for v in per[b])) for b in SIZES]
    out.append("======N={} {} ok={} alpha={:.1f}us step={:.1f}us link={:.1f}GB/s decode{}={:.2f}ms var={:.2f}%---------".format(world, backend, ok, alpha, step, link, args.decode_calls, dec, var))
    if dist.get_rank() == 0:
        for l in out:
            print(l)
        rec = {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "host": "{} ({}x {}, {}, {})".format(socket.gethostname(), world, name, backend, driver()),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cmd": "torchrun --nproc_per_node={} {}".format(world, " ".join(sys.argv)),
            "metrics": {
                "latency_p50_variance_pct": round(var, 2),
                "output_lines": len(out),
                "allreduce_4kb_us": round(mid[0], 1),
                "alpha_us": round(alpha, 1),
                "alpha_step_us": round(step, 1),
                "link_GBps": round(link, 1),
                "decode_allreduce_ms": round(dec, 2),
                "correct": int(ok),
            },
            "runs": {label(b): [round(v, 1) for v in per[b]] for b in SIZES},
            "env": {"torch": torch.__version__, "iters": args.iters, "reps": args.reps, "chunks": args.chunks, "warmup": args.warmup},
            "doc": "bench/results/README.md",
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
