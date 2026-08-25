import json
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_FILES = {
    "cpu": _HERE / "benchmark_results_cpu.json",
    "gpu": _HERE / "benchmark_results_gpu.json",
}
_STYLE = {
    "cpu": {"color": "tab:blue", "marker": "o"},
    "gpu": {"color": "tab:red", "marker": "s"},
}

fig, ax = plt.subplots(figsize=(7, 5))

device_data = {}
for device, path in _FILES.items():
    if not path.exists():
        print(f"Skipping {device}: {path} not found")
        continue
    with open(path) as f:
        data = json.load(f)
    device_data[device] = data
    results = data["results"]
    batch_sizes = [r["batch_size"] for r in results]
    time_per_point = [r["time_per_point_ms"] for r in results]
    label = f"{device.upper()} ({data.get('device_name', 'unknown')})"
    ax.plot(batch_sizes, time_per_point, label=label, **_STYLE[device])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Batch size")
ax.set_ylabel("Time per point (ms)")
ax.set_title("Emulator inference time per point vs. batch size")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

output_path = _HERE / "benchmark_results.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Wrote plot to {output_path}")

if "cpu" in device_data and "gpu" in device_data:
    cpu_by_batch = {r["batch_size"]: r["time_per_point_ms"] for r in device_data["cpu"]["results"]}
    gpu_by_batch = {r["batch_size"]: r["time_per_point_ms"] for r in device_data["gpu"]["results"]}
    common_batches = sorted(set(cpu_by_batch) & set(gpu_by_batch))
    speedup = [cpu_by_batch[b] / gpu_by_batch[b] for b in common_batches]

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(common_batches, speedup, color="tab:green", marker="d")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Speedup (CPU time/point / GPU time/point)")
    ax2.set_title("GPU speedup vs. batch size")
    ax2.grid(True, which="both", alpha=0.3)

    speedup_path = _HERE / "benchmark_speedup.png"
    fig2.savefig(speedup_path, dpi=150, bbox_inches="tight")
    print(f"Wrote speedup plot to {speedup_path}")
else:
    print("Skipping speedup plot: need both cpu and gpu results")
