import random
import time
from math import pi


def compute_pi(num):
    count = 0

    for _ in range(num):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1:
            count += 1

    return 4 * count / num


print("power, res, error_rate, time")

for power in range(10):
    start = time.perf_counter()

    res = compute_pi(10**power)

    runTime = time.perf_counter() - start
    error_rate = abs(res - pi) / pi

    print(power, ", ", res, ", ", error_rate, ", ", runTime)
