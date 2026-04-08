# The "Neural Engine" Benchmark
# 4x4 Matrix-Vector Multiplication

def matvec_mult(matrix, vector, output):
    # Calculate 4 output nodes
    for i in range(4):
        accumulator = 0
        
        # Dot product of row i and the vector
        for j in range(4):
            # The Critical MAC Operation (Multiply-Accumulate)
            # This forces the GNN to balance ALU and Memory timing!
            weight = matrix[i * 4 + j]
            val = vector[j]
            accumulator = accumulator + (weight * val)
            
        output[i] = accumulator