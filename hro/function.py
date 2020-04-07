import math
import numpy as np


def bent_cigar(xs):
    return xs[0] ** 2 + 10 ** 6 * sum([x ** 2 for x in xs[1:]])


def discus(xs):
    return 10 ** 6 * xs[0] ** 2 + sum([x ** 2 for x in xs[1:]])


def exp_ackley(xs):
    d = len(xs)
    p1 = math.sqrt(1.0 / d * sum([x ** 2 for x in xs]))
    p2 = 1.0 / d * sum([math.cos(2 * math.pi * x) for x in xs])

    return -20 * math.exp(-0.2 * p1) - math.exp(p2) + 20 + math.e


def exp_schwefel(xs):
    d = len(xs)
    result = 0
    for i in range(d):
        temp = 0
        for j in range(i):
            temp += xs[j]
        result += temp ** 2
    return result




