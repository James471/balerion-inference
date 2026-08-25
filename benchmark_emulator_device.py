import time
import numpy as np
import tensorflow as tf
from config import load_config, get_models_dir
from pathlib import Path

_ARCH = "32_64_128_256_512_512_512_512_256_128_64_32"
model_path = str(Path(get_models_dir(load_config())) / "uniform" / "reg_arc" / "depth"
                 / _ARCH / "0" / f"model_reg_depth_{_ARCH}_0.keras")

emulator = tf.keras.models.load_model(model_path, compile=False)

for batch_size in [1, 50, 100, 400, 1000, 5000, 20000]:
    x = np.random.uniform(-1, 1, size=(10*batch_size, 10)).astype(np.float32)
    # warmup (first call pays tracing/compilation cost, exclude from timing)
    emulator.predict(x, verbose=0)

    t0 = time.perf_counter()
    for _ in range(10):
        emulator.predict(x, verbose=0)
    t1 = time.perf_counter()
    per_call = (t1 - t0) / 10
    print(f"batch={batch_size:6d}  time/call={1000*per_call:8.3f}ms  "
          f"time/point={1000*per_call/batch_size:8.5f}ms")
