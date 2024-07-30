from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Tuple
import random
from copy import deepcopy
from operator import attrgetter


class abstr_Individ(ABC):
    """
    Structure, that codes solution.
    Includes methods of mutation, mating, Fit evaluating.
    """

    def __init__(self, l_chrom_float: List[float] = None, l_chrom_bin: List[List[int]] = None,
                 f_fit: float = None, f_norm_fit: float = None) -> None:
        self.l_chrom_float = l_chrom_float  # Vector of float values
        self.l_chrom_bin = l_chrom_bin  # Vector of binary values
        self.f_fit = f_fit  # Fitness values
        self.f_norm_fit = f_norm_fit  # Normed fitness values

    # Block for PROCESSING FITNESS
    @abstractmethod
    def _fun_Fit(self) -> float:
        """
        Evaluates Fitness of each individ.
        Should be determined in son-class Individ.

        :return: Value of fitness
        """
        pass

    @staticmethod
    def get_Fit_norm(fit_min: float, fit_max: float, maximization=False) -> Callable:  # Callable[[float], float]
        """
        Returns function for evaluationg normed_fitness from fitness
        according to max and min fitness in population.
        Normed_fitness lies in [0,1]. Uses linear trancformation (y = k*x + b).
        If fit_min == fit_max -> returns function that returns only 1 as normed_fitness.

        :param fit_min: Should be minimal fitness in population
        :param fit_max: Should be maximal fitness in population
        :param maximization: Purpose of optimizing task
        :return: Function, that evaluates normed_fitness from fitness

        Roulette selection operator - one of the most popular selection operators -
        can be specified only for maximization tasks.
        That`s why class GA solves only maximization tasks operating only with normed_fit.
        All minimization tasks should be reduced to maximization by this function.
        """
        ###
        #print(f'In call get_Fit_norm()')
        #print(f'min_fit={fit_min}\tmax_fit={fit_max}')
        #print(f'min_fit==max_fit: {fit_min==fit_max}')
        ###
        if fit_max != fit_min:  # Check whether all individs have same fitness

            # Linear transformation formulas
            if maximization:
                #print(f'fit_max={fit_max}')
                #print(f'fit_min={fit_min}')
                func_fit_norm = lambda fit: 1 / (fit_max - fit_min) * fit - fit_min / (fit_max - fit_min)
            else:
                func_fit_norm = lambda fit: -1 / (fit_max - fit_min) * fit + fit_max / (fit_max - fit_min)

        else:
            ###
            #print(f'Exception detected!')
            ###
            func_fit_norm = lambda fit: 1
        ###
        #print(f'f(0)={func_fit_norm(0)}\tf(1)={func_fit_norm(1)}')
        #print(f'func_fit_norm={func_fit_norm}\n')
        ###
        return func_fit_norm

    @staticmethod
    def _num_b2f(b_chrom: List[int]) -> int:
        """
        Supportive function, translates bin_massive to natural number.

        :param b_chrom: Massive of 0 and 1
        :return: float-value translation of bin massive

        Example: [1, 0, 1, 0] -> 10.0
        """
        f_chrom = 0

        # Iterating through bin massive and evaluating float value
        for i in range(len(b_chrom)):
            f_chrom += b_chrom[i] * 2 ** (len(b_chrom) - i - 1)

        return f_chrom

    @abstractmethod
    def Fit_eval(self, func_fit_norm, eval_norm_fit=True) -> (int, int):
        """
        Evaluates [fitness, normed_fitness] of individ using [_fun_Fit(), func_fit_norm] and
        writes them to individ`s attributes.

        :param func_fit_norm: Function, that evaluates normed_fitness from fitness
        :param eval_norm_fit: Evaluate or not normed_fitness
        :return: (fitness, normed_fitness)
        """
        if self.f_fit == None:
            self.f_fit = self._fun_Fit()  # Evaluating fit

        if eval_norm_fit:
            ###
            #print(f'In @abstractmethod Fit_eval():')
            #print(f'self.__class__={self.__class__}')
            #print(f'func_fit_norm={func_fit_norm}')
            #print(f'f(fit)={func_fit_norm(self.f_fit)}\n')
            #print(f'f(fit)={func_fit_norm(self.f_fit)}\tf(0)={func_fit_norm(0)}\tf(1)={func_fit_norm(1)}\n')
            ###
            self.f_norm_fit = func_fit_norm(self.f_fit)  # Evaluating normed_fit

        return self.f_fit, self.f_norm_fit
    # Block for BIN-FLOAT CHROMOSOME TRANSLATION

    @abstractmethod
    def bin2float(self, l_ran, l_step) -> List[float]:
        """
        Evaluates (list of float_chromosomes) from (list of bin_chromosomes), using _num_b2f(), self.l_chrom_bin and
        writes it to individ`s attribute.

        :param l_ran: List of float_chromosome ranges: [[left_edge1, right_edge1], [left_edge2, right_edge2], ...]
        :param l_step: List of steps of float_chromosome discretization: [step1, step2, ...]
        :return: Float_chromosome of ind: [value1, value2, ...]
        """
        f_ind = []

        # Iterating through each chromosome and evaluating float_chromosomes
        for ran, step, b_chr in zip(l_ran, l_step, self.l_chrom_bin):
            f_ind.append(ran[0] + step * abstr_Individ._num_b2f(b_chr))

        self.l_chrom_float = f_ind

        return f_ind

    @staticmethod
    def _bool_f2b(i_chrom: int, bit_range_chrom: int) -> List[int]:
        """
        Supportive function, translates natural number to bin_massive.

        :param i_chrom: Integer value to transfer
        :param bit_range_chrom: Len of bin_chromosome
        :return: Massive of 0 and 1
        Example: 10 -> [1, 0, 1, 0]
        """
        binary = []

        if i_chrom != 0:
            # Evaluating bin_chromosome from integer value
            for i in range(bit_range_chrom):
                remainder = i_chrom % 2
                binary.append(remainder)
                i_chrom = i_chrom // 2
        else:
            binary = [0] * bit_range_chrom

        return binary[::-1]

    @abstractmethod
    def float2bin(self, l_ran, l_bit_chr) -> List[List[int]]:
        """
        Evaluates (float_chromosome) from  (bin_chromosome) using _bool_f2b(), self.l_chrom_float and
        writes them to individ`s attributes.
        Usually, is not used in GA process.

        :param l_ran: List of float_chromosome ranges: [[left_edge1, right_edge1], [left_edge2, right_edge2], ...]
        :param l_bit_chr: List of bit depths of each chrom: [depth1, depth2, ...]
        :return: List of bin_chromosomes of individ: [[101..1], [110..0], ...]
        """
        b_ind = []

        # Translating each float_chromosome to bin_chromosome
        for ran, bit, f_chr in zip(l_ran, l_bit_chr, self.l_chrom_float):
            # Get integer chromosome to translate to bin_massive
            i_chr = round((f_chr - ran[0]) / (ran[1] - ran[0]) * (2 ** bit - 1))
            b_ind.append(abstr_Individ._bool_f2b(i_chr, bit_range_chrom=bit))

        self.l_chrom_bin = b_ind

        return b_ind

    # Block for EVOLUTION OPERATORS
    @staticmethod
    def _cross_1p(ind1, ind2, self_change=True) -> Tuple:
        """
        Produces 1-point crossover of each binary chromosom-vector between 2 individs.

        :param ind1: 1-st individ
        :param ind2: 2-nd individ
        :param self_change: Create new individ (False; requires extra memory) or operate with previous ones (True)
        :return: A tuple of 2 ind
        """
        # Operate with originals or copies of individs
        if self_change:
            ind_son1 = ind1
            ind_son2 = ind2
        else:
            ind_son1 = deepcopy(ind1)
            ind_son2 = deepcopy(ind2)

        # Crossover proccess over son individs
        size0 = min(len(ind_son1.l_chrom_bin), len(ind_son2.l_chrom_bin))  # Number of chromosomes to crossover
        for chr in range(size0):
            size1 = min(len(ind_son1.l_chrom_bin[chr]), len(ind_son2.l_chrom_bin[chr]))  # Length of chromosome
            cxpoint = random.randint(1, size1 - 1)  # Point of crossover

            # Prevents static right part for 1-st and static left for 2-nd chromosomes
            if random.randint(0, 1):
                ind_son1.l_chrom_bin[chr], ind_son2.l_chrom_bin[chr] = \
                    ind_son2.l_chrom_bin[chr], ind_son1.l_chrom_bin[chr]

            # Swapping right parts of son individs
            ind_son1.l_chrom_bin[chr][cxpoint:], ind_son2.l_chrom_bin[chr][cxpoint:] = \
                ind_son2.l_chrom_bin[chr][cxpoint:], ind_son1.l_chrom_bin[chr][cxpoint:]

        # Reset fit and float_chromosomes of son individs
        ind_son1.f_fit, ind_son1.f_norm_fit, ind_son1.l_chrom_float = None, None, None
        ind_son2.f_fit, ind_son2.f_norm_fit, ind_son2.l_chrom_float = None, None, None

        return ind_son1, ind_son2

    @staticmethod
    def _cross_unif(ind1, ind2, self_change=True) -> Tuple:
        """
        Produces uniform crossover of each binary chromosom-vector between 2 individs.

        :param ind1: 1-st individ
        :param ind2: 2-nd individ
        :param self_change: Create new individ (False; requires extra memory) or operate with previous ones (True)
        :return: A tuple of 2 ind
        """
        # Operate with originals or copies of individs
        if self_change:
            ind_son1 = ind1
            ind_son2 = ind2
        else:
            ind_son1 = deepcopy(ind1)
            ind_son2 = deepcopy(ind2)
        #print(f'f:\n{ind_son1.l_chrom_bin}\n{ind_son2.l_chrom_bin}')

        # Crossover process over son individs
        size0 = min(len(ind_son1.l_chrom_bin), len(ind_son2.l_chrom_bin))  # Number of chromosomes to crossover
        for chr in range(size0):
            size1 = min(len(ind_son1.l_chrom_bin[chr]), len(ind_son2.l_chrom_bin[chr]))  # Length of chromosome
            for bit in range(size1):

                if random.randint(0, 1):
                    ind_son1.l_chrom_bin[chr][bit], ind_son2.l_chrom_bin[chr][bit] = \
                        ind_son2.l_chrom_bin[chr][bit], ind_son1.l_chrom_bin[chr][bit]
                else:
                    ind_son2.l_chrom_bin[chr][bit], ind_son1.l_chrom_bin[chr][bit] = \
                        ind_son2.l_chrom_bin[chr][bit], ind_son1.l_chrom_bin[chr][bit]

        # Reset fit and float_chromosomes of son individs
        ind_son1.f_fit, ind_son1.f_norm_fit, ind_son1.l_chrom_float = None, None, None
        ind_son2.f_fit, ind_son2.f_norm_fit, ind_son2.l_chrom_float = None, None, None

        #print(f's:\n{ind_son1.l_chrom_bin}\n{ind_son2.l_chrom_bin}\n\n')
        return ind_son1, ind_son2

    @staticmethod
    @abstractmethod
    def cross(ind1, ind2, self_change=True) -> List:
        """
        Determines crossover procedure between 2 individs.

        :param ind1: 1-st individ
        :param ind2: 2-nd individ
        :param self_change: Create new individ (False; requares extra memory) or operate with previous ones (True)
        :return: A tuple of 2 ind
        """
        pass

    def mutate_1b(self, self_change=True):
        """
        Inverts one bit in randomly chosen individ`s chromosome.

        :param self_change: Create new ind (False; requires extra memory) or operate with previous ones (True)
        :return: Mutated individ
        """
        # Operate with original or copy of individ
        if self_change:
            ind = self
        else:
            ind = deepcopy(self)

        chr = random.randint(0, len(ind.l_chrom_bin) - 1)  # Choose random bin_chromosome to mutate
        bit_chr = random.randint(0, len(ind.l_chrom_bin[chr]) - 1)  # Choose random bit of bin_chromosome to mutate
        ind.l_chrom_bin[chr][bit_chr] = int(not ind.l_chrom_bin[chr][bit_chr])  # Conduct 1-bit mutation

        # Reset fit and float_chromosomes of individ
        ind.f_fit, ind.f_norm_fit, ind.l_chrom_float = None, None, None

        return ind

    def mutate_1b_LefTri(self, self_change=True):
        """
        Inverts one bit in randomly chosen individ`s chromosome with
        increasing linear probability distribution with bits. (Men`s mutation)

        :param self_change: Create new ind (False; requares extra memory) or operate with previous ones (True)
        :return: Mutated individ
        """
        # Operate with original or copy of individ
        if self_change:
            ind = self
        else:
            ind = deepcopy(self)

        chr = random.randint(0, len(ind.l_chrom_bin) - 1)  # Choose random bin_chromosome to mutate
        bit_chr = int(
            random.triangular(0, len(ind.l_chrom_bin[chr]) - 1, 0))  # Choose random bit of bin_chromosome to mutate
        ind.l_chrom_bin[chr][bit_chr] = int(not ind.l_chrom_bin[chr][bit_chr])  # Conduct 1-bit mutation

        # Reset fit and float_chromosomes of individ
        ind.f_fit, ind.f_norm_fit, ind.l_chrom_float = None, None, None

        return ind

    def mutate_1b_RightTri(self, self_change=True):
        """
        Inverts one bit in randomly chosen individ`s chromosome with
        decreasing linear probability distribution with bits. (Women`s mutation)

        :param self_change: Create new ind (False; requires extra memory) or operate with previous ones (True)
        :return: Mutated individ
        """
        # Operate with original or copy of individ
        if self_change:
            ind = self
        else:
            ind = deepcopy(self)

        chr = random.randint(0, len(ind.l_chrom_bin) - 1)  # Choose random bin_chromosome to mutate
        bit_chr = int(random.triangular(0, len(ind.l_chrom_bin[chr]) - 1,
                                        len(ind.l_chrom_bin[chr]) - 1))  # Choose random bit of bin_chromosome to mutate
        ind.l_chrom_bin[chr][bit_chr] = int(not ind.l_chrom_bin[chr][bit_chr])  # Conduct 1-bit mutation

        # Reset fit and float_chromosomes of individ
        ind.f_fit, ind.f_norm_fit, ind.l_chrom_float = None, None, None

        return ind

    @abstractmethod
    def mutate(self, self_change=True):
        """
        Determines mutation process of individ.

        :param self_change: Create new ind (False; requares extra memory) or operate with previous ones (True)
        :return: Mutated individ
        """
        pass

    # Block for RANDOM GENERATION
    @abstractmethod
    def random_ind(self, l_bit_chr):
        """
        Creating individs with random chromosoms.
        Function of instance of class, but do not uses any parametres of it.

        :param l_bit_chr: List of bit depths of each chrom: [depth1, depth2, ...]
        :return: Individ
        """
        #print(f'\n{l_bit_chr}\n')
        #print(f'{range_chrom_float}\n')
        b_ind = []

        # Iterating through list of depths and creating random chromosomes
        for ran in l_bit_chr:
            # Evaluate random quantity number of 0 and 1
            zero_count = random.randint(0, ran)
            one_count = ran - zero_count

            # Creating random chromosome
            chr = [0] * zero_count + [1] * one_count
            random.shuffle(chr)
            b_ind.append(chr)

        ind = self.__class__(l_chrom_bin=b_ind)  # Creating individ
        ind.bin2float()  # Eval float_chromosoms
        ind.Fit_eval(eval_norm_fit=False)  # Evaluate fit

        return ind


class abstr_Population(ABC):
    """
    Structure, that codes solution. Includes methods of selection.
    Transfers from abstr_Individ Crossover, Mating, Fit evaluating to list of ind.
    """

    def __init__(self, l_ind: List[abstr_Individ] = [], l_elite_ind: List[abstr_Individ] = [],
                 f_mut_prob=None, f_cross_prob=1) -> None:
        self.individs = l_ind  # List of normal individs
        self.elite_individs = l_elite_ind  # List of elite individs
        self.f_cross_prob = f_cross_prob  # Probability of crossover between 2 individs in Cross()
        self.f_mut_prob = f_mut_prob  # Probability of mutation individ in Mut()

    # Block for PROCESSING FITNESS
    def borderFit(self, with_elite=True) -> Tuple[float, float]:
        """
        Finds min/max fit in population. Needed for get_Fit_norm().

        :param with_elite: Include elite individs or not
        :return: Min/max fit
        """
        # Create list of individs
        if with_elite:
            l_ind = self.individs + self.elite_individs
        else:
            l_ind = self.individs

        # Create list of fits of individs in population
        l_fit = [ind.f_fit for ind in l_ind]

        return min(l_fit), max(l_fit)

    def selBorder(self, k=1, with_elite=True, max_fit=True, deepcopy_ind=True) -> List[abstr_Individ]:
        """
        Select k best individs from Population according to f_fit.

        :param k: The number of individs to select
        :param with_elite: Including elite individs
        :param max_fit: Purpose of optimizing task
        :param deepcopy_ind: Returns individs as new objects
        or references on current individs from self.individs and self.elite_individs
        :returns: A list containing the k best individs
        """
        # Create list of individs
        if with_elite:
            l_ind = self.individs + self.elite_individs
        else:
            l_ind = self.individs

        # Get k best individs and return them
        if deepcopy_ind:
            return sorted(deepcopy(l_ind), key=attrgetter('f_fit'), reverse=max_fit)[:k]
        else:
            return sorted(l_ind, key=attrgetter('f_fit'), reverse=max_fit)[:k]

    def _selRoulette(self, k=1, deepcopy_ind=True) -> List[abstr_Individ]:
        """
        Select k individuals from the Total Population (Normal+Elite ind)
        according to f_norm_fit using fortune Roulette idea.

        :param k: The number of individuals to select
        :returns: A list of copies/refers of selected ind
        """
        # Function for coping objects
        if deepcopy_ind:
            my_copy = lambda x: deepcopy(x)
        else:
            my_copy = lambda x: x

        l_pop_ind = self.individs + self.elite_individs  # All (rural and elite) individs in population

        # Process of Roulette selection
        s_inds = sorted(l_pop_ind, key=attrgetter('f_norm_fit'), reverse=True)  # Sorting individs in population
        sum_fits = sum(ind.f_norm_fit for ind in l_pop_ind)  # Total sum of fits of individs in population
        chosen = []

        for i in range(k):
            u = random.random() * sum_fits  # Evaluate random accumulative fit for roulatte select

            sum_ = 0
            # Find individ with accumulated fit contains random fit
            for ind in s_inds:
                sum_ += ind.f_norm_fit
                if sum_ > u:
                    chosen.append(my_copy(ind))
                    break


        return chosen

    def _selTournament(self, tournsize: int, k=1, deepcopy_ind=True, normed_fit=True, maximization=True) -> \
            List[abstr_Individ]:
        """
        Select the best individual among tournsize randomly chosen
        individuals of Total Population (Normal+Elite ind), k times.

        :param tournsize: The number of individuals participating in each tournament
        :param k: The number of individuals to select
        :param normed_fit: Using normed fitness or "raw" fitness to sort
        :param maximization: Purpose of optimizing task.
        If using normed fitness (f_norm_fit) for Select() -> maximization = True
        If do not use normed fitness (f_norm_fit) for Select() in minimisation task -> maximization = False
        :returns: A list of copies/refers of selected ind
        """

        def _selRandom(individs, k):
            """
            Select k individuals at random from the input individuals with
            replacement.
            The returned list contains references to the input individs.

            :param individs: List of individs to select
            :param k: Number individs to select
            :return: List of selected individs
            """
            return [random.choice(individs) for i in range(k)]

        # Function for coping objects
        if deepcopy_ind:
            my_copy = lambda x: deepcopy(x)
        else:
            my_copy = lambda x: x

        l_pop_ind = self.individs + self.elite_individs  # All (rural and elite) individs in population
        chosen = []

        # Process of Tournament selection
        for i in range(k):
            aspirants = _selRandom(l_pop_ind, tournsize)  # Individs participating in Tournament

            # Usual case
            if (normed_fit, maximization) == (True, True):
                chosen.append(my_copy(max(aspirants, key=attrgetter('f_norm_fit'))))

            # Could be used for tasks without normed fitness
            elif (normed_fit, maximization) == (False, True):
                chosen.append(my_copy(max(aspirants, key=attrgetter('f_fit'))))

            # Could be used for tasks without normed fitness
            elif (normed_fit, maximization) == (False, False):
                chosen.append(my_copy(min(aspirants, key=attrgetter('f_fit'))))

            # Wrong case: normed fitness should transform any kind of task to maximization.
            # But anyway, let it be
            elif (normed_fit, maximization) == (True, False):
                chosen.append(my_copy(min(aspirants, key=attrgetter('f_norm_fit'))))

        return chosen

    '''
    def _selRoulette_pairs(self, k=1, deepcopy_ind=True) -> List[abstr_Individ]:
        """
        Select k/2 individual`s pairs from the Total Population (Normal+Elite ind)
        according to f_norm_fit using fortune Roulette idea.
        One individ can't be selected twice one pair.

        :param k: The number of individuals to select
        :returns: A list of copies/refers of selected ind
        """
        # Function for coping objects
        if deepcopy_ind:
            my_copy = lambda x: deepcopy(x)
        else:
            my_copy = lambda x: x

        l_pop_ind = self.individs + self.elite_individs  # All (rural and elite) individs in population

        # Process of Roulette selection
        s_inds = sorted(l_pop_ind, key=attrgetter('f_norm_fit'), reverse=True)  # Sorting individs in population
        sum_fits = sum(ind.f_norm_fit for ind in l_pop_ind)  # Total sum of fits of individs in population
        chosen = []

        for i in range(k):
            u = random.random() * sum_fits  # Evaluate random accumulative fit for roulatte select

            sum_ = 0
            # Find individ with accumulated fit contains random fit
            for i_i in range(len(s_inds)):
                sum_ += s_inds[i_i].f_norm_fit
                if sum_ > u:
                    chosen.append(my_copy(s_inds[i_i]))
                    break


        return chosen
        
    '''

    @abstractmethod
    def Select(self, k=1) -> List[abstr_Individ]:
        """
        Should be determined in son-class Population.

        :param k: The number of individs to select
        :return: A list of copies of selected individs
        """
        pass  # Should choose 1 of Selection strategeis

    def update_Fit(self, eval_norm_fit=True) -> None:
        """
        Search ind with no fit=None and evaluates fit and norm_fit.

        :param eval_norm_fit: Eval norm_fit or not
        :return: None
        """
        l_ind = self.individs + self.elite_individs
        for ind in l_ind:
            '''
            if ind.f_fit == None:  # If ind fit is not stated - eval it and, if eval_norm_fit==True, eval norm_fit
                ind._fun_Fit(eval_norm_fit)
            elif eval_norm_fit:  # If fit is evaluated and norm_fit==None - eval norm_fit
                ind.f_norm_fit(eval_norm_fit)
            '''
            ind.Fit_eval(eval_norm_fit)

    # Block for BIN-FLOAT CHROMOSOME TRANSLATION
    def Bin2Float(self) -> None:
        """
        Evaluates float_chromosomes from bin_chromosome for each individ with
        (float_chromosome != None) and (bin_chromosome = None);
        Arise bin2float() from class abstr_Individ.

        :return: None
        """
        for ind in self.individs + self.elite_individs:
            if ind.l_chrom_bin != None and ind.l_chrom_float == None:
                ind.bin2float()

    def Float2Bin(self) -> None:
        """
        Evaluates bin_chromosome from float_chromosomes for each individ with
        (bin_chromosome = None) and (float_chromosome != None);
        Arise float2bin() from class abstr_Individ.

        :return: None
        """
        for ind in self.individs + self.elite_individs:
            if ind.l_chrom_float != None and ind.l_chrom_bin == None:
                ind.float2bin()

    # Block for EVOLUTION OPERATORS
    @staticmethod
    def Cross(l_ind1: List[abstr_Individ], l_ind2: List[abstr_Individ], f_cross_prob: float, self_change=True) -> \
            Tuple[List[abstr_Individ], List[abstr_Individ]]:
        """
        Provides abstr_Individ.cross() at each pair of individs in l_ind1, l_ind2.
        Can be provided at individs from different Ind/Pop classes.
        Cross function will be taken from individs of 1-st list.

        :param f_cross_prob: Probability of crossing (usually 1)
        :param l_ind2: List of 1-st individs from each crossing pair
        :param l_ind1: List of 2-nd individs from each crossing pair
        :param self_change: Create new individs (False; requires extra memory) or operate with previous ones (True)
        :return: 2 lists of new ind - copies.
        """
        if len(l_ind1) != len(l_ind2):  # Checking lists of ind have the same length
            ###
            print(f'l1={len(l_ind1)}\t l2={len(l_ind2)}')
            #
            #print('\n\n\nl_ind1:\n')
            #for i_i, i_ind in zip(range(len(l_ind1)), l_ind1):
            #    print(f'{i_i} ind from l_ind1')
            #    i_ind.my_print()
            #
            #for i_i, i_ind in zip(range(len(l_ind2)), l_ind2):
            #    print(f'{i_i} ind from l_ind2')
            #    i_ind.my_print()
            ###
            raise Exception('l_ind1 and l_ind2 has different length')
        l_ind_son1, l_ind_son2 = [], []
        for ind1, ind2 in zip(l_ind1, l_ind2):
            if random.random() < f_cross_prob:  # If random strikes - mate inds
                ind1, ind2 = ind1.cross(ind1, ind2, self_change)
            l_ind_son1.append(ind1)
            l_ind_son2.append(ind2)

        return l_ind_son1, l_ind_son2

    def Mut(self, self_change=True) -> List[abstr_Individ]:
        """
        Provides mutation over all Normal ind in Population.

        :param self_change: Create new ind (False; requires extra memory) or operate with previous ones (True)
        :return: Pop with mutated ind
        """
        l_ind = []
        for ind in self.individs:
            if random.random() < self.f_mut_prob:  # If random strikes - mute ind
                ind = ind.mutate(self_change)
            l_ind.append(ind)
        return l_ind

    # Block for RANDOM GENERATION
    @abstractmethod
    def Random_generation(self,
                          num_ind: int, num_elite_ind: int,
                          f_mut_prob: float, f_cross_prob: float,
                          class_Ind):
        """
        Generates random population of individs and chooses elite from them.
        Function of instance of class, but do not use any parameters of it.
        """
        l_ind = [class_Ind().random_ind() for i in range(num_ind)]  # Random generating Normal ind
        l_elite_ind = [class_Ind().random_ind() for i in range(num_ind)]  # Random generating Elite ind

        pop = self.__class__(l_ind=l_ind, l_elite_ind=l_elite_ind,
                             f_mut_prob=f_mut_prob, f_cross_prob=f_cross_prob)

        # Bin2Float
        # pop.Bin2Float()

        # Fit eval
        pop.update_Fit(eval_norm_fit=False)

        # Choose elite
        pop.elite_individs = pop.selBorder(k=num_elite_ind)

        return pop
