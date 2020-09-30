import numpy as np
from ea import function
from ea.Individual import Individual_pso
import matplotlib.pyplot as plt
import random
import copy

class PSO(object):
    '''
    这里做的是函数优化，如果是处理二进制的问题需要进行修改calculate_fit里面的适应度函数
    '''

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
        self.global_fitness = 9999999 if optimal_minimal else 0
        self.global_vector = []
        self.best_fitness_store = []
        self.fitness_fun = function.exp_ackley

    def init_stage(self):
        for i in range(self.size):
            individual = Individual_pso(self.n, self.n_bit, self.upper, self.lower)
            self.population.append(individual)

    def fit(self):
        # 开始初始化：
        self.init_stage()
        # 计算适应度值
        self.append_fitness()
        # 和全局最优进行比较
        self.compare_with_best()

        # 开始主要的循环：
        for i in range(self.gen):

            # 开始更新：
            self.update()

            self.append_fitness()
            self.compare_with_best()
            # 收集每一次迭代之后的最优值
            self.best_fitness_store.append(self.current_best_fitness)

    def update(self):
        current_best = self.population[self.current_best_index]
        w = random.uniform(0.4, 0.9)
        c1 = 2
        c2 = 2
        for i in range(self.size):
            if i != self.current_best_index:
                for j in range(self.n * self.n_bit):
                    # 更新速度
                    part1 = c1 * random.random() * (self.global_vector[j] - self.population[i].vector[j])
                    part2 = c2 * random.random() * (current_best.vector[j] - self.population[i].vector[j])
                    self.population[i].velocity[j] = w * self.population[i].velocity[j] + part1 + part2
                    # 更新位置
                    s = 1 / (1 + (np.exp(-self.population[i].velocity[j])))
                    if random.random() <= s:
                        self.population[i].vector[j] = 1
                    else:
                        self.population[i].vector[j] = 0

    def calculate_fit(self, individual):
        '''
        处理二进制问题的时候主要是修改这里！！！
        :param individual: 种群中的每个个体
        :return:返回的是适应度值
        '''
        # 计算每个个体的适应度：
        gene_dec = []
        for j in range(self.n_bit - 1, self.n * self.n_bit, self.n_bit):
            # 将一个个体的二进制基因串拆分成十进制基因中对应的维度：
            k = j - self.n_bit + 1
            gene_dec.append(self.decode(individual.vector[k:j]))
        return self.fitness_fun(np.array(gene_dec))

    def decode(self, gene):
        # 将进制进行转换：
        length = len(gene)
        dec = 0
        for j in gene:
            dec = dec + int(j) * (2 ** (length - 1))
            length = length - 1
        # 转换到规定的范围中：
        gene_dec = dec * (self.upper - self.lower) * 1.0 / (2 ** len(gene) - 1) + self.lower
        return gene_dec

    def append_fitness(self):
        self.current_fitness_store = [self.calculate_fit(self.population[i]) for i in range(self.size)]
        self.current_best_fitness = min(self.current_fitness_store) if self.optimal_minimal \
            else max(self.current_fitness_store)
        self.current_best_index = self.current_fitness_store.index(self.current_best_fitness)

    def compare_with_best(self):
        # 这里只写了求最小值时候的比较，比较最大值的时候再加
        if self.optimal_minimal and self.current_best_fitness < self.global_fitness:
            self.global_fitness = self.current_best_fitness
            self.global_vector = self.population[self.current_best_index].vector


pso = PSO(60, 5, 15, 10, -10, 1000, True)
pso.fit()
plt.plot(np.arange(1, 1000 + 1), [v for v in pso.best_fitness_store])
plt.xlabel('Gen')
plt.ylabel('f(x)')
plt.show()