import json
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import load_config, get_models_dir

_ARCH = "32_64_128_256_512_512_512_512_256_128_64_32"
model_path = str(Path(get_models_dir(load_config())) / "uniform" / "reg_arc" / "depth"
                 / _ARCH / "0" / f"model_reg_depth_{_ARCH}_0.keras")

gpus = tf.config.list_physical_devices('GPU')
use_gpu = len(gpus) > 0
device = '/GPU:0' if use_gpu else '/CPU:0'

print(f"GPUs visible to TensorFlow: {gpus}")
print(f"Running on: {device}")

if use_gpu:
    device_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'unknown GPU')
else:
    device_name = platform.processor() or platform.machine()
print(f"Device name: {device_name}")

with tf.device(device):
    emulator = tf.keras.models.load_model(model_path, compile=False)

    results = []
    for batch_size in [1, 50, 100, 400, 1000, 5000]:
        n_points = 10 * batch_size
        x = np.random.uniform(-1, 1, size=(n_points, 10)).astype(np.float32)
        # warmup (first call pays tracing/compilation cost, exclude from timing)
        emulator.predict(x, verbose=0)

        t0 = time.perf_counter()
        for _ in range(10):
            emulator.predict(x, verbose=0)
        t1 = time.perf_counter()
        per_call = (t1 - t0) / 10
        per_point = per_call / n_points
        print(f"batch={batch_size:6d}  time/call={1000*per_call:8.3f}ms  "
              f"time/point={1000*per_point:8.5f}ms")
        results.append({
            "batch_size": batch_size,
            "n_points": n_points,
            "time_per_call_ms": 1000 * per_call,
            "time_per_point_ms": 1000 * per_point,
        })

output = {
    "device": "gpu" if use_gpu else "cpu",
    "device_name": device_name,
    "gpus_visible": [g.name for g in gpus],
    "results": results,
}

output_path = Path(__file__).parent / f"benchmark_results_{'gpu' if use_gpu else 'cpu'}.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Wrote results to {output_path}")
