# Complexity: Extreme - The Matrix Multiply "Holy Grail"
# We'll use a 2x2 version to keep the DAG manageable but the logic intact

# Input A (2x2)
a00, a01 = 1, 2
a10, a11 = 3, 4

# Input B (2x2)
b00, b01 = 5, 6
b10, b11 = 7, 8

# Row 0, Col 0
c00_1 = a00 * b00
c00_2 = a01 * b10
c00 = c00_1 + c00_2

# Row 0, Col 1
c01_1 = a00 * b01
c01_2 = a01 * b11
c01 = c01_1 + c01_2

# Row 1, Col 0
c10_1 = a10 * b00
c10_2 = a11 * b10
c10 = c10_1 + c10_2

# Row 1, Col 1
c11_1 = a10 * b01
c11_2 = a11 * b11
c11 = c11_1 + c11_2
