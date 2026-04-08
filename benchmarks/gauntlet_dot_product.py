# Gauntlet 2: Dot Product
# Demonstrates a Multiply-Accumulate (MAC) chain.
# Teaches the agent to handle the long dependency chain of the reduction tree.

def gauntlet_dot_product(A, B):
    accumulator = 0
    for i in range(4):
        # MUL (latency 3) happens first, then ADD (latency 1)
        val = A[i] * B[i]
        accumulator = accumulator + val
    return accumulator
