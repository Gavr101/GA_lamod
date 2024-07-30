import os
import sys
from typing import List
from numpy import e, pi, exp, cos, sqrt, sum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from GA_kernal_lib.Ind_kernal import Individ
from GA_kernal_lib.Pop_kernal import Population
from GA_kernal_lib.GA_kernal import class_NGA

import os


RANGE_Chrom_Float = [[-100, 100]] * 32
BIT_Chrom = [25] * 32

MAXIMIZATION = True


def _FUN_FIT(self) -> int:
    """Task of approximation"""

    def ackley(individual):
        """
        Ackley test objective function.
        """
        N = len(individual)
        return 20 - 20 * exp(-0.2 * sqrt(1.0 / N * sum(x ** 2 for x in individual))) \
            + e - exp(1.0 / N * sum(cos(2 * pi * x) for x in individual))

    return 20 + e - ackley(self.l_chrom_float)



class Individ_NGA(Individ):
    range_chrom_float = RANGE_Chrom_Float
    bit_chrom = BIT_Chrom

    # Determination of referenced function from father-class Individ
    @staticmethod
    def _get_step_chrom(range_chrom_float, bit_chrom):
        """Supportive function to determine the step_chrom."""
        return Individ._get_step_chrom(range_chrom_float, bit_chrom)

    step_chrom = _get_step_chrom.__func__(range_chrom_float, bit_chrom)

    maximization = MAXIMIZATION

    def _fun_Fit(self) -> int:
        """Task of approximation"""
        return _FUN_FIT(self)


class Population_NGA(Population):
    class_Ind = Individ_NGA



# -== Providing experiments ==-

dim = 32
func_name = 'ackley'
Num_generation_max = 8000

for n in range(1, 101):
    DIR = f"Exp__NGA_correct_criter/{dim}_dim/ackley"
    ncss_file_name = f"/ncss_{n}.txt"

    if not os.path.isfile(DIR + ncss_file_name):  # Проверка существования директории
        # print(DIR+ncss_file_name, os.path.isfile(DIR+ncss_file_name))

        NGA = class_NGA(class_Pop=Population_NGA,
                        num_ind=94,
                        num_elite_ind=6,
                        f_mut_prob=0.01,
                        num_epoch_to_stop=100,
                        delta_fit=0.001,
                        log_FDir=DIR,
                        log_FName=str(n),
                        left_epoch_bound=Num_generation_max,
                        Min_logging=True,
                        Max_logging=False)

        NGA.Launch()

    else:
        print(f'DIR: {DIR + ncss_file_name} - skipped')
