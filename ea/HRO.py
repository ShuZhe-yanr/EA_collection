import numpy as np
from ea import function
from ea.Individual import Individual_hro
import matplotlib.pyplot as plt
import random


class HRO(object):
    def __init__(self, size, n, n_bit, upper, lower, gen, optimal_minimal):
        self.size = size
        self.n = n
        self.n_bit = n_bit
        self.upper = upper
        self.lower = lower
        self.gen = gen
        self.optimal_minimal = optimal_minimal
        self.population = []
        self.current_fitness_store = []
        self.current_best_index = 0
        self.current_best_fitness = 0
        self.best_fitness_store = []
        self.fitness_fun = function.exp_ackley

    def amend(self, value):
        '''
        过界之后需要调整
        :param value:
        :param upper:
        :param lower:
        :return:
        '''
        if value > self.upper:
            return self.upper
        elif value < self.lower:
            return self.lower
        else:
            return value


    def compare(self, old_s, new_s):
        '''
        比较新生成的和之前的个体之间fitness
        :param old_s:
        :param new_s:
        :return:
        '''
        old_fitness = self.calculate_fit(old_s.vector, 'dec')
        new_fitness = self.calculate_fit(new_s.vector, 'dec')
        if (self.optimal_minimal and new_fitness < old_fitness) or (not self.optimal_minimal and new_fitness > old_fitness):
            return new_s, 1
        else:
            return old_s, -1

    def fit(self):
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
        # 划分三个系
        part = int(self.size / 3)

        # 初始化种群：
        population = []
        for i in range(self.size):
            individual = Individual_hro(self.n, self.upper, self.lower)
            population.append(individual)

        # 开始迭代：
        for i in range(self.gen):
            # 首先对可行解进行排序：从小到大
            population.sort(key=lambda s: self.calculate_fit(s.vector, 'dec'))
            # 判断如果是求最大值，就需要反转：
            if not self.optimal_minimal:
                population.reverse()

            # hybridization stage
            for j in range(2*part, self.size):
                # 生成一个temp用来存储新生成的个体
                temp_solution = Individual_hro(self.n, self.upper, self.lower)
                for k in range(self.n):
                    r1 = random.uniform(-1, 1)
                    r2 = random.uniform(-1, 1)
                    sterile_index = random.randint(2*part, self.size-1)
                    maintainer_index = random.randint(0, part)
                    temp_solution.vector[k] = (r1 * population[sterile_index].vector[k] + r2 *
                                               population[maintainer_index].vector[k]) / (r1 + r2)
                    self.amend(temp_solution.vector[k])
                population[j], _ = self.compare(population[j], temp_solution)

            # selfing stage and renewal stage
            for j in range(part, 2*part):
                temp_solution = Individual_hro(self.n, self.upper, self.lower)
                if not population[j].is_exceed_trial():
                    for k in range(self.n):
                        restorer_index = random.randint(part, 2*part)
                        r3 = random.random()
                        temp_solution.vector[k] = r3 * (population[0].vector[k] - population[restorer_index].vector[k]) + \
                                                  population[j].vector[k]
                        self.amend(temp_solution.vector[k])
                    population[j], lost = self.compare(population[j], temp_solution)
                    if lost == -1:
                        population[j].trial_increase()
                    else:
                        population[j].trial_zero()

                else:
                    # renewal stage:
                    for k in range(self.n):
                        r4 = random.random()
                        temp_solution.vector[k] = population[j].vector[k] + r4 * (self.upper - self.lower) + self.lower
                        self.amend(temp_solution.vector[k])
                    population[j], lost = self.compare(population[j], temp_solution)



            # 计算fitness:
            self.current_fitness_store = [self.calculate_fit(s.vector, 'dec') for s in population]
            self.current_best_fitness = min(self.current_fitness_store) if self.optimal_minimal else \
                max(self.current_fitness_store)
            current_best_index = self.current_fitness_store.index(self.current_best_fitness)

            self.best_fitness_store.append(self.current_best_fitness)

    def calculate_fit(self, vector, str):
        '''
        这里是转换函数，将十进制算法转换成二进制
        :param vector: 十进制向量
        :param str: 控制进制转换的字符串
        :return: 返回适应度值
        '''
        if str == 'binary':
            binary = []
            for i in range(self.n):
                s = 1 / (1 + (np.exp(-vector[i])))
                if random.random() <= s:
                    binary.append(1)
                else:
                    binary.append(0)
            return self.fitness_fun(binary)
        else:
            return self.fitness_fun(vector)


hro = HRO(60, 5, 15, 10, -10, 1000, True)
hro.fit()
plt.plot(np.arange(1, 1000 + 1), [v for v in hro.best_fitness_store])
plt.xlabel('Gen')
plt.ylabel('f(x)')
plt.show()







