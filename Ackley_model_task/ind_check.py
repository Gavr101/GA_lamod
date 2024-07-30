from NGA_ackley32 import Population_NGA, Individ_NGA
from GA_kernal_lib.GA_kernal import class_NGA


ind_0 = Individ_NGA()
ind_1 = ind_0.random_ind()

print(ind_1.bit_chrom)
print(ind_1.l_chrom_bin, '\n',  ind_1.l_chrom_float, '\n', ind_1.step_chrom)
print(len(ind_1.l_chrom_bin), len(ind_1.l_chrom_float))