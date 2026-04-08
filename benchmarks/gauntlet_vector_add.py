# Gauntlet 1: Vector Add
# Demonstrates pure horizontal parallelism.
# A 2-ALU constraint should take 2 cycles for math, 4 for loads.

def gauntlet_vector_add(A, B, C):
    for i in range(4):
        # Independent operations
        C[i] = A[i] + B[i]
