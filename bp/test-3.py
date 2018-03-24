#!/usr/bin/python
#coding=utf-8
#see http://garfileo.is-programmer.com/2011/2/19/hello-ga.24563.html
import math, random
import matplotlib.pyplot as plt
import numpy as np  

class Population:
    def __init__ (self, size, chrom_size, cp, mp, gen_max):
        # 种群信息
        self.individuals = []          # 个体集合
        self.fitness = []              # 个体适应度集合
        self.selector_probability = [] # 个体选择概率集合
        self.new_individuals = []      # 新一代个体集合
         
        self.elitist = {'chromosome':[0, 0], 'fitness':0, 'age':0} # 最佳个体的信息
         
        self.size = size # 种群所包含的个体数
        self.chromosome_size = chrom_size # 个体的染色体长度
        self.crossover_probability = cp   # 个体之间的交叉概率
        self.mutation_probability = mp    # 个体之间的变异概率
         
        self.generation_max = gen_max # 种群进化的最大世代数
        self.age = 0                  # 种群当前所处世代
        self.maxdat = []
        self.mindat = []
        # 随机产生初始个体集，并将新一代个体、适应度、选择概率等集合以 0 值进行初始化
        v = 2 ** self.chromosome_size - 1
        for i in range (self.size):
            self.individuals.append ([random.randint (0, v), random.randint (0, v)])
            self.new_individuals.append ([0, 0])
            self.fitness.append (0)
            self.selector_probability.append (0)

# 接上面的代码
	"""
		基于轮盘赌博机的选择

		decode 函数可以将一个染色体 chromosome 映射为区间 interval 之内的数值。精确到小数点后6为

		fitness_func 是适应度函数，可以根据个体的两个染色体计算出该个体的适应度，这里直接采用了本文所要求解的目标函数：
		
		evaluate 函数用于评估种群中的个体集合 self.individuals 中各个个体的适应度，
		即将各个个体的 2 个染色体代入 fitness_func 函数，并将计算结果保存在 self.fitness 列表中，
		然后将 self.fitness 中的各个个体适应度除以所有个体适应度之和，得到各个个体的生存概率。
		为了适合轮盘赌博游戏，需要将个体的生存概率进行叠加，从而计算出各个个体的选择概 率。
	"""
    def decode (self, interval, chromosome):
        d = interval[1] - interval[0]
        n = float (2 ** self.chromosome_size -1)
        return (interval[0] + chromosome * d / n)
     
    def fitness_func (self, chrom1, chrom2):
        interval = [-10.0, 10.0]
        (x, y) = (self.decode (interval, chrom1), 
                  self.decode (interval, chrom2))
        n = lambda x, y: math.sin (math.sqrt (x*x + y*y)) ** 2 - 0.5
        d = lambda x, y: (1 + 0.001 * (x*x + y*y)) ** 2
        func = lambda x, y: 0.5 - n (x, y)/d (x, y)
        return func (x, y)
         
    def evaluate (self):
        sp = self.selector_probability
        for i in range (self.size):
            self.fitness[i] = self.fitness_func (self.individuals[i][0], 
                                                 self.individuals[i][1])
        ft_sum = sum (self.fitness)
        for i in range (self.size):
            sp[i] = self.fitness[i] / float (ft_sum)
        for i in range (1, self.size):
            sp[i] = sp[i] + sp[i-1]
            # 接上面的代码
    def select (self):
        (t, i) = (random.random (), 0)
        for p in self.selector_probability:
            if p > t:
                break
            i = i + 1
        return i

    def reproduct_elitist (self):
        # 与当前种群进行适应度比较，更新最佳个体
        j = 0
        for i in range (self.size):
            if self.elitist['fitness'] < self.fitness[i]:
                j = i
                self.elitist['fitness'] = self.fitness[i]
        if (j > 0):
            self.elitist['chromosome'][0] = self.individuals[j][0]
            self.elitist['chromosome'][1] = self.individuals[j][1]
            self.elitist['age'] = self.age
        # 接上面的代码
        '''
         cross 函数可以将两个染色体进行交叉配对，从而生成 2 个新染色体。

		此 处使用染色体交叉方法很简单，先生成一个随机概率 p，
		如果两个待交叉的染色体不同并且 p 小于种群个体之间的交叉概率 self.crossover_probability，
		那么就在 中间随机选取一个位置，将两个染色体分别断为 2 截
        '''
    def cross (self, chrom1, chrom2):
        p = random.random ()
        n = 2 ** self.chromosome_size -1
        if chrom1 != chrom2 and p < self.crossover_probability:
            t = random.randint (1, self.chromosome_size - 1)
            mask = n << t
            (r1, r2) = (chrom1 & mask, chrom2 & mask)
            mask = n >> (self.chromosome_size - t)
            (l1, l2) = (chrom1 & mask, chrom2 & mask)
            (chrom1, chrom2) = (r1 + l2, r2 + l1)
        return (chrom1, chrom2)
        # 接上面代码
        '''
        mutate 函数可以将一个染色体按照变异概率进行单点变异
        '''
    def mutate (self, chrom):
        p = random.random ()
        if p < self.mutation_probability:
            t = random.randint (1, self.chromosome_size)
            mask1 = 1 << (t - 1)
            mask2 = chrom & mask1
            if mask2 > 0:
                chrom = chrom & (~mask2)
            else:
                chrom = chrom ^ mask1
        return chrom

        # 接上面的代码
        """
         evolve 函数可以实现种群的一代进化计算，计算过程分为三个步骤：

	    使用 evaluate 函数评估当前种群的适应度，并计算各个体的选择概率。
	    对 于数量为 self.size 的 self.individuals 集合，循环 self.size / 2 次，
	    每次从 self.individuals 中选出 2 个个体，对其进行交叉和变异操作，
	    并将计算结果保存于新的个体集合 self.new_individuals 中。
	    用种群进化生成的新个体集合 self.new_individuals 替换当前个体集合。

        """
    def evolve (self):
        indvs = self.individuals
        new_indvs = self.new_individuals
         
        # 计算适应度及选择概率
        self.evaluate ()
         
        # 进化操作
        i = 0
        while True:
            # 选择两个个体，进行交叉与变异，产生新的种群
            idv1 = self.select ()
            idv2 = self.select ()
             
            # 交叉
            (idv1_x, idv1_y) = (indvs[idv1][0], indvs[idv1][1])
            (idv2_x, idv2_y) = (indvs[idv2][0], indvs[idv2][1])
            (idv1_x, idv2_x) = self.cross (idv1_x, idv2_x)
            (idv1_y, idv2_y) = self.cross (idv1_y, idv2_y)
             
            # 变异
            (idv1_x, idv1_y) = (self.mutate (idv1_x), self.mutate (idv1_y))
            (idv2_x, idv2_y) = (self.mutate (idv2_x), self.mutate (idv2_y))
             
            (new_indvs[i][0], new_indvs[i][1])     = (idv1_x, idv1_y)
            (new_indvs[i+1][0], new_indvs[i+1][1]) = (idv2_x, idv2_y)
             
            # 判断进化过程是否结束
            i = i + 2
            if i >= self.size:
                break
         # 最佳个体保留
        self.reproduct_elitist () 
        # 更新换代
        for i in range (self.size):
            self.individuals[i][0] = self.new_individuals[i][0]
            self.individuals[i][1] = self.new_individuals[i][1]
            # 接上面的代码

            
    def run (self):
        for i in range (self.generation_max):
            self.evolve ()
            self.maxdat.append(max (self.fitness))
            self.mindat.append(min (self.fitness))
            # print (i, max (self.fitness), sum (self.fitness)/self.size, 
            #        min (self.fitness))

    def display(self):
        print (self.elitist)
    	plt.figure(figsize=(8,4)) #创建绘图对象
    	plt.plot(self.maxdat)   
        plt.plot(self.mindat) 
    	plt.legend(('max','min'),loc='best')
    	plt.show()  #显示图 


            # 接上面的代码
if __name__ == '__main__':
    # 种群的个体数量为 50，染色体长度为 25，交叉概率为 0.8，变异概率为 0.1,进化最大世代数为 150
    pop = Population (50, 24, 0.8, 0.5, 150)
    pop.run()
    pop.display()