from typing import List

from GA_kernal_lib.abstr_Ind_Pop import abstr_Individ


class Individ(abstr_Individ):
    """
    Practical realisation of individ. Refers to abstr_Individ.

    Determines: maximization -                  Max/Min optimization problem;
                range_chrom_float, bit_chrom -  Space of search and it`s discretization;
                _fun_Fit -                      Function for fitness evaluation;
                my_print() -                    Print characteristics of print.

    Transfers (hidden):
                get_Fit_norm();
                _num_b2f(); - determines relation between pheno- and geno- types
                _bool_f2b(); -  --**--
                cross();
                mutate().

    Transfers:
                bin2float();
                float2bin();
                random_ind().

    Other methods can be overloaded.
    """

    range_chrom_float = [[0, 31], [0, 31]]  # Range of each float_chromosome
    bit_chrom = [5, 5]  # Bit depth of each bit_chromosome

    @staticmethod
    def _get_step_chrom(range_chrom_float, bit_chrom):
        """
        Supportive function to determine the step_chrom.

        :param range_chrom_float: Range of each float_chromosome
        :param bit_chrom: Bit depth of each bit_chromosome
        :return: Discretization of float chrom values
        """
        step_chrom = []  # Discretization of float chrom values
        for ran, bit in zip(range_chrom_float, bit_chrom):
            step_chrom.append((ran[1] - ran[0]) / (2 ** bit - 1))
        return step_chrom

    step_chrom = _get_step_chrom.__func__(range_chrom_float, bit_chrom)


    maximization = True  # Purpose of optimization problem
    func_fit_norm = lambda x: None  # Evaluate normed fitness from raw fitness
    # Need to be updated each epoch by Individ.get_Fit_norm. Do not need to be determined

    def _fun_Fit(self) -> int:
        """One-max optimization function"""
        s = 0
        for arr in self.l_chrom_bin:
            s += sum(arr)
        return s

    # Determination of referenced function from father-class abstr_Individ
    def Fit_eval(self, eval_norm_fit) -> (int, int):
        ###
        #print('In Fit_eval:')
        #print(f'self.__class__={self.__class__}')
        #print(f'self.__class__.func_fit_norm={self.__class__.func_fit_norm}\n')
        #print(f'f(0)={self.__class__.func_fit_norm(0)}\tf(1)={self.__class__.func_fit_norm(1)}\n')
        ###
        return super().Fit_eval(self.__class__.func_fit_norm, eval_norm_fit)

    def bin2float(self):
        return super().bin2float(self.__class__.range_chrom_float,
                                 self.__class__.step_chrom)

    def float2bin(self):
        return super().float2bin(self.__class__.range_chrom_float,
                                 self.__class__.bit_chrom)

    @staticmethod
    def cross(ind1, ind2, self_change=True) -> List:
        return ind1.__class__._cross_1p(ind1, ind2, self_change=self_change)

    def mutate(self, self_change=True) -> List:
        return super().mutate_1b(self_change=self_change)

    def random_ind(self):
        #print(f'\n{self.__class__.bit_chrom}\n')
        #print(f'\n{self.__class__.range_chrom_float}\n')
        return super().random_ind(self.__class__.bit_chrom)

    def my_print(self):
        '''
        Print characteristics of individ.
        '''
        print(f'f_chr:\t{self.l_chrom_float}')
        print(f'b_chr:\t{self.l_chrom_bin}')
        print(f'fit:\t{self.f_fit}')
        print(f'norm_fit:\t{self.f_norm_fit}')


# bin2float() and float2bin() Check
'''
ind = Individ().random_ind()
#ind = Individ(l_chrom_bin = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
ind.my_print()
print(0)

ind.bin2float()
ind.my_print()
print(1)


ind.l_chrom_bin = None
ind.my_print()
print(2)

ind.float2bin()
ind.my_print()
print(3)
'''

# cross(ind1, ind2) Check
'''
self_change = True
ind1 = Individ(l_chrom_bin=[[1,1,1,1,1],[1,1,1,1,1]])
ind2 = Individ(l_chrom_bin=[[0,0,0,0,0],[0,0,0,0,0]])

print('input:')
ind1.my_print()
ind2.my_print()
print()

s1,s2 = Individ.cross(ind1, ind2, self_change = self_change)
print('self')
ind1.my_print()
ind2.my_print()
print()

print('output:')
s1.my_print()
s2.my_print()
'''

# mutate(ind) Check
'''
self_change = False
ind = Individ(l_chrom_bin=[[1,1,1,1,1],[1,1,1,1,1]])

print('input:')
ind.my_print()
print()

ind1 = ind.mutate(self_change = self_change)
print('self:')
ind.my_print()
print()
print('output:')
ind1.my_print()
'''

# Fit_eval() and func_fit_norm Check
'''
eval_norm_fit = True

ind = Individ(l_chrom_bin=[[1,0,1,1,1],[1,1,1,0,1]])
ind.my_print()
print()

ind.f_fit = None
ind.my_print()
print()

Individ.func_fit_norm = Individ.get_Fit_norm(1, 9, Individ.maximization)
ind.Fit_eval(eval_norm_fit = eval_norm_fit)
ind.my_print()
'''
