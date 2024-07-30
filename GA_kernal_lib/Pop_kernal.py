from typing import List

from GA_kernal_lib.abstr_Ind_Pop import abstr_Population
from GA_kernal_lib.Ind_kernal import Individ


class Population(abstr_Population):
    """
    Practical realisation of population. Refers to father-class abstr_population.

    Determines: Select() -  Selection method (from functions from abstr_Population);
                class_Ind - Class of Individ;

    Transfers:  Random_generation();

    Other methods can be overloaded.
    """

    class_Ind = Individ

    def Select(self, k=1, deepcopy_ind=True) -> List[Individ]:
        return super()._selRoulette(k=k, deepcopy_ind=deepcopy_ind)

    def Random_generation(self,
                          num_ind: int, num_elite_ind: int,
                          f_mut_prob: float, f_cross_prob=1):
        return super().Random_generation(num_ind=num_ind, num_elite_ind=num_elite_ind,
                                         f_mut_prob=f_mut_prob, f_cross_prob=f_cross_prob,
                                         class_Ind=self.class_Ind)

    def my_print(self, all_ind=True):
        """
        Print each individ`s characteristics in population.
        """
        print(f'Norm ind:\t{len(self.individs)}')
        print(f'Elite ind:\t{len(self.elite_individs)}')
        if all_ind:
            print('Norm inds:')
            print('N\tfloat_chr\tbin_chr\t\tfit\tnorm_fit')
            for i, ind in zip(range(1, len(self.individs) + 1), self.individs):
                print(f'{i}. {ind.l_chrom_float}\t{ind.l_chrom_bin}\t{ind.f_fit}\t{ind.f_norm_fit}')
            print()
            print('Elite inds:')
            print('N\tfloat_chr\tbin_chr\t\tfit\tnorm_fit')
            for i, ind in zip(range(1, len(self.elite_individs) + 1), self.elite_individs):
                print(f'{i}. {ind.l_chrom_float}\t{ind.l_chrom_bin}\t{ind.f_fit}\t{ind.f_norm_fit}')
        else:
            print('1-st norm ind:')
            self.individs[0].my_print()
            print()
            print(f'1-st elite ind:')
            self.elite_individs[0].my_print()
        print()
        print(f'Cross prob-ty:\t{self.f_cross_prob}')
        print(f'Mut prob-ty:\t{self.f_mut_prob}')


# Random generation, update_Fit, Selection Check
"""
# Random generation Check
print('----- Random generation Check ------')
print()

pop = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)
pop.my_print(all_ind = True)

# update_Fit Check
print()
print('----- update_Fit Check ------')
print()
fit_min = pop.borderFit(max_fit = False)
fit_max = pop.borderFit(max_fit = True)
fit_min, fit_max = pop.borderFit()

pop.class_Ind.func_fit_norm = pop.class_Ind.get_Fit_norm(fit_min=fit_min, fit_max=fit_max, maximization = True)
pop.update_Fit(eval_norm_fit = True)
pop.my_print(all_ind = True)


# Selection Check
print()
print('----- Selection Check ------')
print()
#print(pop.Select(k = 3))
l_ind = pop.Select(5)
l_elite_ind = pop.selBorder(2, max_fit = True)
pop1 = Population(l_ind = l_ind, l_elite_ind = l_elite_ind,
                       f_mut_prob = pop.f_mut_prob, f_cross_prob = pop.f_cross_prob)
pop1.my_print(all_ind = True)

'''
# del pop with referenced ind Check
print()
print('----- del pop with referenced ind Check ------')
print()
ind = pop.individs[0]
del pop
pop1.my_print(all_ind = True)
ind.my_print()
'''
"""

# Cross
'''
print()
print('----- Cross Check ------')
print()
pop1 = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)
pop2 = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)

pop1.my_print()
pop2.my_print()

print()
print('After Cross:')
print()

# Cross
l_ind1 = pop1.individs
l_ind2 = pop2.individs
l_ind_son1, l_ind_son2 = Population.Cross(l_ind1 = l_ind1, l_ind2 = l_ind2, f_cross_prob = pop1.f_cross_prob)

# Creating new pop
pop_son1 = Population(l_ind = l_ind_son1, f_mut_prob = pop1.f_mut_prob, f_cross_prob = pop1.f_cross_prob)
pop_son2 = Population(l_ind = l_ind_son2, f_mut_prob = pop2.f_mut_prob, f_cross_prob = pop2.f_cross_prob)

pop_son1.my_print()
pop_son2.my_print()
'''

# Mutation
'''
print()
print('----- Cross Check ------')
print()
pop = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)
pop.my_print()

print()
print('After Mut:')
print()

pop.Mut()
pop.my_print()
'''

# Selection + Cross + Mut Check
# Will be transfered to GA class
'''
print()
print('----- Selection + Cross + Mut Check ------')
print()
pop1 = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)
pop2 = Population.Random_generation(num_ind=5, num_elite_ind = 2, f_mut_prob = 0.01)


pop1.my_print()
pop2.my_print()

# Norm fit eval
fit_min1, fit_max1 = pop1.borderFit()
fit_min2, fit_max2 = pop2.borderFit()
fit_min = min(fit_min1, fit_min2)
fit_max = max(fit_max1, fit_max2)
pop1.class_Ind.func_fit_norm = pop1.class_Ind.get_Fit_norm(fit_min=fit_min, fit_max=fit_max, maximization = True)
pop1.update_Fit(eval_norm_fit = True)
pop2.update_Fit(eval_norm_fit = True)

print()
print('----- After norm fit eval ------')
print()
pop1.my_print()
pop2.my_print()

# Preliminarily forming son_pop for coping elite ind
pop_son1 = Population(l_elite_ind = deepcopy(pop1.elite_individs), f_mut_prob = pop1.f_mut_prob, f_cross_prob = pop1.f_cross_prob)
pop_son2 = Population(l_elite_ind = deepcopy(pop2.elite_individs), f_mut_prob = pop2.f_mut_prob, f_cross_prob = pop2.f_cross_prob)

print()
print('Preliminarily forming son_pop:')
print()
pop_son1.my_print()
pop_son2.my_print()

# Selection
l_ind_sel1, l_ind_sel2 = pop1.Select(len(pop1.individs)), pop2.Select(len(pop2.individs))

# Cross
l_ind_son1, l_ind_son2 = Population.Cross(l_ind1 = l_ind_sel1, l_ind2 = l_ind_sel2, f_cross_prob = pop1.f_cross_prob)

pop_son1.individs  = l_ind_son1
pop_son2.individs = l_ind_son2

print()
print('After Selection and Cross:')
print()
pop_son1.my_print()
pop_son2.my_print()

# Mut
pop_son1.Mut()
pop_son2.Mut()

pop_son1.Bin2Float()
pop_son2.Bin2Float()

print()
print('After Mutation and Bin2Float:')
print()
pop_son1.my_print()
pop_son2.my_print()


# new Norm fit eval
pop_son1.update_Fit(eval_norm_fit = False)
pop_son2.update_Fit(eval_norm_fit = False)

fit_min = min(pop_son1.borderFit(max_fit = False), pop_son2.borderFit(max_fit = False))
fit_max = max(pop_son1.borderFit(max_fit = True), pop_son2.borderFit(max_fit = True))
pop_son1.class_Ind.func_fit_norm = pop_son1.class_Ind.get_Fit_norm(fit_min=fit_min, fit_max=fit_max, maximization = True)

pop_son1.update_Fit(eval_norm_fit = True)
pop_son2.update_Fit(eval_norm_fit = True)

print()
print('After new Norm fit eval:')
print()
pop_son1.my_print()
pop_son2.my_print()

# Forming new elite ind
pop_son1.elite_individs = pop_son1.selBorder(k = len(pop_son1.elite_individs), deepcopy_ind = False)
pop_son2.elite_individs = pop_son2.selBorder(k = len(pop_son2.elite_individs), deepcopy_ind = False)

print()
print('After forming new elite ind:')
print()
pop_son1.my_print()
pop_son2.my_print()
'''
