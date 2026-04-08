# Complexity: Peak - Tests port bottlenecks and data reuse logic
p1, p2, p3, p4, p5, p6, p7, p8, p9 = 1, 2, 3, 4, 5, 6, 7, 8, 9

# X-direction gradient (Reusable data access)
gx_1 = p3 + p6
gx_2 = p6 + p9
gx_pos = gx_1 + gx_2

gx_3 = p1 + p4
gx_4 = p4 + p7
gx_neg = gx_3 + gx_4

gx_final = gx_pos - gx_neg

# Y-direction gradient (More reuse)
gy_1 = p1 + p2
gy_2 = p2 + p3
gy_pos = gy_1 + gy_2

gy_3 = p7 + p8
gy_4 = p8 + p9
gy_neg = gy_3 + gy_4

gy_final = gy_pos - gy_neg
