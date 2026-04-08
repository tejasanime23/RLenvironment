# Gauntlet 4: The "Trap" Script
# Demonstrates dynamic logic that is UN-SYNTHESIZABLE into hardware.
# This proves the environment correctly identifies software-vs-hardware limits.

def gauntlet_trap(data):
    # DYNAMIC WHILE LOOP: The compiler cannot know how many clock cycles 
    # to allocate because the loop trip count depends on the runtime value of data[0].
    # This will trigger a graceful HLSSynthesisError.
    
    i = 0
    while data[i] > 0:
        data[i] = data[i] - 1
        i = (i + 1) % 4
    
    return data
