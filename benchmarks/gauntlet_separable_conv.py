# Complexity: High - Tests global synchronization points
# Phase 1: Depthwise (Highly Parallel - 4 operations)
dw0 = 10 * 1
dw1 = 20 * 2
dw2 = 30 * 3
dw3 = 40 * 4

# Phase 2: Pointwise (Strict Reduction - Single Chain)
# Pointwise cannot start until ALL dw nodes are finished
pw_temp1 = dw0 + dw1
pw_temp2 = dw2 + dw3
out = pw_temp1 + pw_temp2
