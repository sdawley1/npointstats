#!/bin/bash
import numpy as np
from numba import cuda


@cuda.jit
def increment_by_one(an_array: np.array) -> None:
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1
    return


if __name__ == "__main__":

    N = int(2**24)
    A = np.random.rand(N)

    threadsperblock = 32
    blockspergrid = (A.size + (threadsperblock - 1)) // threadsperblock
    increment_by_one[blockspergrid, threadsperblock](A)
