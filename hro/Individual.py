import random
import numpy as np


class Individual(object):
    TRIAL_LIMIT = 8

    def __init__(self, n, max, min):
        self.vector = []
        self.n = n
        self.max = max
        self.min = min
        self.trial = 0
        self.create()

    def create(self):
        U = np.array([self.max] * self.n)
        L = np.array([self.min] * self.n)
        random_diag = np.diag(np.random.uniform(0, 1, self.n))
        self.vector = self.min + np.matmul(random_diag, U - L)

    def trial_increase(self):
        self.trial += 1

    def trial_zero(self):
        self.trial = 0

    def is_exceed_trial(self):
        return self.trial >= Individual.TRIAL_LIMIT