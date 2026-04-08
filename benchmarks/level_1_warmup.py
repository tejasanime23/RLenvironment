# Level 1: The Warmup (Simple Arithmetic)
# Teaches the agent to map independent AST nodes to parallel ALUs
# Because there are no data dependencies between x and y, 
# a 2-ALU environment should schedule both operations on the exact same clock cycle!

def warmup_kernel(a, b, c, d):
    # These two operations are parallel!
    x = a + b
    y = c * d
    
    return x, y
