# Level 3: The Constraint (Memory Bottlenecks)
# This forces the GNN to balance ALU speeds against a strict 1-MEM data-port timing limit!
# Arrays force sequential LOADs, destroying MakeSpan unless Pragmas (Array Partition) are used.

def matvec_mult(matrix, vector, output):
    # Calculate 4 output nodes
    for i in range(4):
        accumulator = 0
        
        # Dot product of row i and the vector
        for j in range(4):
            # The Critical MAC Operation (Multiply-Accumulate)
            weight = matrix[i * 4 + j]
            val = vector[j]
            accumulator = accumulator + (weight * val)
            
        output[i] = accumulator
