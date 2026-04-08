# Complexity: High - Tests non-linear dependency paths and cross-over scheduling
x0, x1, x2, x3 = 10, 20, 30, 40

# Stage 1: Initial Cross (Parallel)
s1_0 = x0 + x2
s1_1 = x1 + x3
s1_2 = x0 - x2
s1_3 = x1 - x3

# Stage 2: Second Cross (Interleaved Dependencies)
# Every node here depends on two different nodes from Stage 1
y0 = s1_0 + s1_1
y1 = s1_2 + s1_3
y2 = s1_0 - s1_1
y3 = s1_2 - s1_3
