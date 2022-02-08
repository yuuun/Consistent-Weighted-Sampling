import numpy as np
class RandomNumberGenerator():
    def __init__(self, dim, n_sig):
        self.ran_r1 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_r2 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_c1 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_c2 = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]
        self.ran_b = [np.random.uniform(0, 1, dim + 1) for _ in range(n_sig)]