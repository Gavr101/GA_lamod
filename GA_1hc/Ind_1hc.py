import numpy as np
from typing import List

from GA_kernal_lib.Ind_kernal import Individ


class Individ_1hc(Individ):
    range_chrom_float = [[None, None], ]
    bit_chrom = [10]

    @staticmethod
    def _num_b2f(b_chrom):
        pass

    @staticmethod
    def _bool_f2b(i_chrom, bit_range_chrom):
        pass

    def bin2float(self):
        pass

    def float2bin(self):
        pass

    maximization = False

    # target_chrom = np.array([int(i%12==False) for i in range(bit_chrom[0])])
    target_chrom = np.array([1 for i in range(bit_chrom[0])])

    def _fun_Fit(self) -> int:
        """Task of approximation of binarized chromosome"""
        s = 0
        for arr in self.l_chrom_bin:
            s += sum((np.array(arr) - Individ_1hc.target_chrom) ** 2)
        return s

    @staticmethod
    def cross(ind1, ind2, self_change=True) -> List:
        return ind1.__class__._cross_1p(ind1, ind2, self_change=self_change)

    def mutate(self, self_change=True) -> List:
        return super().mutate_1b(self_change=self_change)