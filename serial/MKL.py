import numpy as np
import time

N = 3000 #matrix size
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

C = A @ B # warm-up

t0 = time.perf_counter()
C = A @ B
t1 = time.perf_counter()

gflops = (2 * N**3) / (t1 - t0) / 1e9
print(f"Time: {t1 - t0:.3f} s")
print(f"Performance: {gflops:.2f} GFLOPS")