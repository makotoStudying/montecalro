from __future__ import print_function, absolute_import

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import time
from math import pi


@cuda.jit
def compute_pi(rng_states, iterations, out):
    """Find the maximum value in values and store in result[0]"""
    thread_id = cuda.grid(1)

    # Compute pi by drawing random (x, y) points and finding what
    # fraction lie inside a unit circle
    inside = 0
    for _ in range(iterations):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x**2 + y**2 <= 1.0:  # type: ignore
            inside += 1

    out[thread_id] = 4.0 * inside / iterations


threads_per_block = 512
blocks = 128
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = np.zeros(threads_per_block * blocks, dtype=np.float32)


for power in range(12):
    iterations = -(-(10**power) // (threads_per_block * blocks))

    start = time.perf_counter()

    compute_pi[blocks, threads_per_block](rng_states, iterations, out)

    runTime = time.perf_counter() - start

    res = out.mean()

    error_rate = abs(res - pi) / pi

    if power == 0:
        print("power, res, error_rate, time")

    print(power, ", ", res, ", ", error_rate, ", ", runTime)
