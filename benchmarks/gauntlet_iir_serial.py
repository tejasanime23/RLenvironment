# Complexity: Medium - Tests the agent's ability to handle zero parallelism
# Total Depth: 8 Levels
val = 100
a = val + 1
b = a * 2   # Depends on a
c = b + 3   # Depends on b
d = c * 4   # Depends on c
e = d + 5   # Depends on d
f = e * 6   # Depends on e
g = f + 7   # Depends on f
h = g * 8   # Depends on g
