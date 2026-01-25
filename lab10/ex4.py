import numpy as np

def get_roots(vec_polynomial_coefficients):
    n = len(vec_polynomial_coefficients)
    
    C = np.zeros((n,n))

    if n > 1:
        for i in range(n - 1):
            C[i + 1, i] = 1

    for i in range(n):
        C[i, n - 1] = -vec_polynomial_coefficients[i]

    roots = np.linalg.eigvals(C)

    return roots