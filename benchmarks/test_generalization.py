# Verification of Generalization Patches
def generalization_kernel(a, b):
    # Tests += (AugAssign)
    acc = 0
    acc += a
    acc += b
    
    # Tests UnaryOp
    neg_acc = -acc
    
    # Tests BoolOp
    if a > 0 and b > 0:
        neg_acc = neg_acc * 2
        
    return neg_acc
