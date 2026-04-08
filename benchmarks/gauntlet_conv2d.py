# Gauntlet 3: 2D Convolution Snippet
# Demonstrates nested loops and complex 2D spatial indexing.
# Tests if the compiler can handle complex address translation logic.

def gauntlet_conv2d(image, kernel, output):
    # Simplified 3x3 window Conv
    for i in range(2):
        for j in range(2):
            acc = 0
            # Inner window
            for ki in range(2):
                for kj in range(2):
                    pixel = image[(i + ki) * 4 + (j + kj)]
                    weight = kernel[ki * 2 + kj]
                    acc = acc + (pixel * weight)
            output[i * 2 + j] = acc
