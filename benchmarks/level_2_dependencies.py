# Level 2: The Wall (Data Dependencies)
# Teaches the agent about Pipeline Stalls!
# Even if the environment has 999 ALUs, it cannot run these in parallel.
# The graph physically forces chronological waiting.

def sequential_dependency_kernel(a, b, c, d, e):
    # Step 1
    x = a + b
    
    # Step 2 (MUST wait for x)
    y = x * c
    
    # Step 3 (MUST wait for y)
    z = y - d
    
    # Step 4 (MUST wait for z)
    final_output = z / e
    
    return final_output
