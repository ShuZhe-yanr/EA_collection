import numpy as np
import function
from Individual import Individual
import matplotlib.pyplot as plt
import random


def amend(value, upper, lower):
    '''
    过界之后需要调整
    :param value:
    :param upper:
    :param lower:
    :return:
    '''
    if value > upper:
        return upper
    elif value < lower:
        return lower
    else:
        return value


def compare(old_s, new_s, optimal_minimal, fitness_func):
    '''
    比较新生成的和之前的个体之间fitness
    :param old_s:
    :param new_s:
    :return:
    '''
    old_fitness = fitness_func(old_s.vector)
    new_fitness = fitness_func(new_s.vector)
    if (optimal_minimal and new_fitness < old_fitness) or (not optimal_minimal and new_fitness > old_fitness):
        return new_s, 1
    else:
        return old_s, -1


def hro(size, n, upper, lower, gen, optimal_minimal):
    '''
    主函数
    :param size:种群数
    :param n: 维度
    :param upper: 最大值
    :param lower: 最小值
    :param gen: 迭代次数
    :param optimal_minimal:求最大值还是最小值
    :return: 输出的是每代最优值，最后用来画图
    '''
    # 获取适应度函数：
    fitness_func = function.exp_ackley
    # 划分三个系
    part = int(size / 3)
    # 每代的全局最优
    best_fitness_store = []

    # 初始化种群：
    population = []
    for i in range(size):
        individual = Individual(n, upper, lower)
        population.append(individual)

    # 开始迭代：
    for i in range(gen):
        # 首先对可行解进行排序：从小到大
        population.sort(key=lambda s: fitness_func(s.vector))
        # 判断如果是求最大值，就需要反转：
        if not optimal_minimal:
            population.reverse()

        # hybridization stage
        for j in range(2*part, size):
            # 生成一个temp用来存储新生成的个体
            temp_solution = Individual(n, upper, lower)
            for k in range(n):
                r1 = random.uniform(-1, 1)
                r2 = random.uniform(-1, 1)
                sterile_index = random.randint(2*part, size-1)
                maintainer_index = random.randint(0, part)
                temp_solution.vector[k] = (r1 * population[sterile_index].vector[k] + r2 *
                                           population[maintainer_index].vector[k]) / (r1 + r2)
                amend(temp_solution.vector[k], upper, lower)
            population[j], _ = compare(population[j], temp_solution, optimal_minimal, fitness_func)

        # selfing stage and renewal stage
        for j in range(part, 2*part):
            temp_solution = Individual(n, upper, lower)
            if not population[j].is_exceed_trial():
                for k in range(n):
                    restorer_index = random.randint(part, 2*part)
                    r3 = random.random()
                    temp_solution.vector[k] = r3 * (population[0].vector[k] - population[restorer_index].vector[k]) + \
                                              population[j].vector[k]
                    amend(temp_solution.vector[k], upper, lower)
                population[j], lost = compare(population[j], temp_solution, optimal_minimal, fitness_func)
                if lost == -1:
                    population[j].trial_increase()
                else:
                    population[j].trial_zero()

            else:
                # renewal stage:
                for k in range(n):
                    r4 = random.random()
                    temp_solution.vector[k] = population[j].vector[k] + r4 * (upper - lower) + lower
                    amend(temp_solution.vector[k], upper, lower)
                population[j], lost = compare(population[j], temp_solution, optimal_minimal, fitness_func)

        # 计算fitness:
        current_fitness_store = [fitness_func(s.vector) for s in population]
        current_best_fitness = min(current_fitness_store) if optimal_minimal else \
            max(current_fitness_store)
        current_best_index = current_fitness_store.index(current_best_fitness)

        best_fitness_store.append(current_best_fitness)
    return best_fitness_store


if __name__ == '__main__':
    store = hro(60, 10, 10, -10, 1000, True)
    plt.plot(np.arange(1, 1000 + 1), [v for v in store])
    plt.xlabel('Gen')
    plt.ylabel('f(x)')
    plt.show()







