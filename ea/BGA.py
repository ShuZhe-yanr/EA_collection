import numpy as np
from ea import function
from ea.Individual import Individual_ga
import matplotlib.pyplot as plt
import random
import copy

class GA(object):
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
        self.best_fitness_store = []
        self.fitness_fun = function.exp_ackley

    def fit(self):
        for i in range(self.size):
            individual = Individual_ga(self.n, self.n_bit, self.upper, self.lower)
            self.population.append(individual)

        self.append_fitness()

        # 开始迭代：
        for i in range(self.gen):
            # 首先需要进行轮盘赌：
            binary_string = self.roulette()
            # 替换原来population中
            for j in range(self.size):
                self.population[j].vector = copy.copy(binary_string[j])

            # 开始变异
            for j in range(self.size):
                if j != self.current_best_index:
                    index = random.randint(0, self.n*self.n_bit-1)
                    rand = random.random()

                    if rand < 0.05:
                        self.population[j].vector[index] = 1 if random.random() > 0.5 else 0

            # 开始交叉;
            for j in range(0, self.size, 2):
                # 依条件需要交叉的分量下标
                index = random.randint(0, self.n*self.n_bit-1)
                rand_probability = random.random()

                if rand_probability < 0.8:
                    self.population[j].vector[index], self.population[j + 1].vector[index] = \
                        self.population[j + 1].vector[index], self.population[j].vector[index]

            #
            self.append_fitness()
            self.best_fitness_store.append(self.current_best_fitness)

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
        '''
        在处理十进制的问题的时候，将二进制的字符串转换成十进制数
        :param gene: 二进制的字符串
        :return: 返回的是二进制对应的十进制数
        '''
        # 将进制进行转换：
        length = len(gene)
        dec = 0
        for j in gene:
            dec = dec + int(j) * (2 ** (length - 1))
            length = length - 1
        # 转换到规定的范围中：
        gene_dec = dec * (self.upper - self.lower) * 1.0 / (2 ** len(gene) - 1) + self.lower
        return gene_dec

    def roulette(self):
        roulette_list = copy.copy(self.current_fitness_store)

        # 对求最小值和最大值进行一个判断
        if self.optimal_minimal:
            # 如果是求最小值，就将数值倒过来
            for i, val in enumerate(roulette_list):
                if roulette_list[i] != 0:
                    roulette_list[i] = 1 / roulette_list[i]
        current_fitness_sum = sum(roulette_list)

        temp_binary = []
        # 进行选择
        for i in range(self.size):
            if i != self.current_best_index:
                rand = random.uniform(0, current_fitness_sum)
                temp_roulette_rate = 0.0
                temp_j = 0

                for j, val in enumerate(roulette_list):
                    temp_roulette_rate += val

                    if rand < temp_roulette_rate:
                        temp_j = j
                        break

                # 将第i次循环选中的个体，赋值给第i个个体
                temp_binary.append(self.population[temp_j].vector)
            else:
                temp_binary.append(self.population[i].vector)

        return temp_binary

    def append_fitness(self):
        self.current_fitness_store = [self.calculate_fit(self.population[i]) for i in range(self.size)]
        self.current_best_fitness = min(self.current_fitness_store) if self.optimal_minimal \
            else max(self.current_fitness_store)
        self.current_best_index = self.current_fitness_store.index(self.current_best_fitness)


ga = GA(60, 5, 15, 10, -10, 1000, True)
ga.fit()
plt.plot(np.arange(1, 1000 + 1), [v for v in ga.best_fitness_store])
plt.xlabel('Gen')
plt.ylabel('f(x)')
plt.show()