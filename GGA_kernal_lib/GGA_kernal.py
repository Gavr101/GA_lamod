from copy import deepcopy
import time
import random
from typing import List
from operator import attrgetter

from GA_kernal_lib.Logging_kernal import CSV_Log_agent, JSON_Create


class class_GGA:
    """
    Gender GA.
    Determines process of evolution, 3-stage logging
    """
    name = "GGA"
    l_timing_names = ['Copy elites', 'SEL', 'CROSS', 'MUT', 'Fit eval', 'Choose elites']

    def __init__(self, class_Pop_M,
                 class_Pop_F,
                 num_ind: int, num_elite_ind: int,
                 f_mut_prob_M: float,
                 f_mut_prob_F: float,
                 num_epoch_to_stop: int, delta_fit: float,
                 log_FDir: str, log_FName: str,
                 left_epoch_bound=0, right_epoch_bound=float("inf"),
                 f_cross_prob=1,
                 Min_logging=True, Max_logging=False, print_pop=False) \
            -> None:
        self.class_Pop_M = class_Pop_M  # Classes of Males` and Females` Individs must be the same
        self.class_Ind_M = class_Pop_M.class_Ind  # for ability to cross them

        self.class_Pop_F = class_Pop_F
        self.class_Ind_F = class_Pop_F.class_Ind

        # Input main characteristics of GA
        # Genderised characteristics
        self.num_ind_M, self.num_ind_F = num_ind // 2, num_ind // 2
        self.num_elite_ind_M, self.num_elite_ind_F = num_elite_ind // 2, num_elite_ind // 2
        self.f_mut_prob_M, self.f_mut_prob_F = f_mut_prob_M, f_mut_prob_F
        self.f_cross_prob = f_cross_prob

        # General characteristics
        self.num_epoch_to_stop = num_epoch_to_stop
        self.delta_fit = delta_fit
        self.left_epoch_bound = left_epoch_bound
        self.right_epoch_bound = right_epoch_bound

        # Input logging characteristics of GA
        self.log_FDir = log_FDir
        self.log_FName = log_FName
        self.Min_logging = Min_logging
        self.Max_logging = Max_logging
        self.print_pop = print_pop

        # Supportive parameters
        self.max_fit_M, self.max_fit_F = None, None
        self.min_fit_M, self.min_fit_F = None, None
        self.best_fit = None
        self.best_ind = None

        self.num_generation = 0
        self.epoch_static_fit = 0

    def Preiteration(self) -> None:
        """
        Preparative process before main evolution.
        Generate random population and evaluate it`s parameters
        """

        # Random generation of population
        self.pop_M = self.class_Pop_M().Random_generation(num_ind=self.num_ind_M,
                                                          num_elite_ind=self.num_elite_ind_M,
                                                          f_mut_prob=self.f_mut_prob_M,
                                                          f_cross_prob=self.f_cross_prob)

        self.pop_F = self.class_Pop_F().Random_generation(num_ind=self.num_ind_F,
                                                          num_elite_ind=self.num_elite_ind_F,
                                                          f_mut_prob=self.f_mut_prob_F,
                                                          f_cross_prob=self.f_cross_prob)

        self.pop_M.Bin2Float()
        self.pop_F.Bin2Float()

        # Update supportive parameters
        self.min_fit_M, self.max_fit_M = self.pop_M.borderFit()
        self.min_fit_F, self.max_fit_F = self.pop_F.borderFit()
        # self.min_fit, self.max_fit = min(min_M, min_F), max(max_M, max_F)

        if self.class_Ind_M.maximization:
            self.best_fit = max(self.max_fit_M, self.max_fit_F)
        else:
            self.best_fit = min(self.min_fit_M, self.min_fit_F)

        # Update norm Fit and remember best fit
        self.class_Ind_M.func_fit_norm = self.class_Ind_M.get_Fit_norm(fit_min=self.min_fit_M,
                                                                       fit_max=self.max_fit_M,
                                                                       maximization=self.class_Ind_M.maximization)
        self.class_Ind_F.func_fit_norm = self.class_Ind_F.get_Fit_norm(fit_min=self.min_fit_F,
                                                                       fit_max=self.max_fit_F,
                                                                       maximization=self.class_Ind_F.maximization)

        self.pop_M.update_Fit(eval_norm_fit=True)
        self.pop_F.update_Fit(eval_norm_fit=True)

        print('\n\n\n----\nInit')
        if self.print_pop:
            print('Males:')
            print(f'min_fit_M={self.min_fit_M}\tmax_fit_M={self.max_fit_M}')
            self.pop_M.my_print()
            print('= = = = =')
            print('Females:')
            print(f'min_fit_F={self.min_fit_F}\tmax_fit_F={self.max_fit_F}')
            self.pop_F.my_print()

    def Iteration(self, timing_eval=True) -> List[float]:
        """
        Main process of evolution. Applied iteratively.
        Conducts:   Selection
                    Crossover
                    Mutation
                    Update of population`s parameters
        """
        # For timing evaluating
        l_timing = []
        ###
        #print(f'On curr iter:')
        #print(f'min_fit_M={self.min_fit_M}\tmax_fit_M={self.max_fit_M}')
        #print(f'min_fit_F={self.min_fit_F}\tmax_fit_F={self.max_fit_F}\n')
        ###
        if self.print_pop:
            print('Males:')
            self.pop_M.my_print()
            print('= = = = =')
            print('Females:')
            self.pop_F.my_print()

        # Preliminarily forming son_pop for coping elite individs
        l_timing.append(time.perf_counter())  # Timing (0->)

        pop_son_M = self.class_Pop_M(l_elite_ind=deepcopy(self.pop_M.elite_individs),
                                     f_mut_prob=self.pop_M.f_mut_prob,
                                     f_cross_prob=self.pop_M.f_cross_prob)
        pop_son_F = self.class_Pop_F(l_elite_ind=deepcopy(self.pop_F.elite_individs),
                                     f_mut_prob=self.pop_F.f_mut_prob,
                                     f_cross_prob=self.pop_F.f_cross_prob)

        # SELECTION
        l_timing.append(time.perf_counter())  # Timing (->1->)

        l_ind_sel_M = self.pop_M.Select(self.num_ind_M, deepcopy_ind=False)
        l_ind_sel_F = self.pop_F.Select(self.num_ind_F, deepcopy_ind=False)

        # CROSSOVER
        l_timing.append(time.perf_counter())  # Timing (->2->)

        l_ind_son_M, l_ind_son_F = self.class_Pop_M.Cross(l_ind1=l_ind_sel_M,
                                                          l_ind2=l_ind_sel_F,
                                                          f_cross_prob=self.pop_M.f_cross_prob,
                                                          self_change=False)

        pop_son_M.individs = l_ind_son_M
        pop_son_F.individs = l_ind_son_F

        # MUTATION
        l_timing.append(time.perf_counter())  # Timing (->3->)

        pop_son_M.Mut()
        pop_son_F.Mut()

        # Assign pop_son to self.individs!!!
        # self.pop.individs = pop_son

        # Evaluate float_chromosomes from bin_chromosomes using Bin2Float()
        l_timing.append(time.perf_counter())  # Timing (->4)

        pop_son_M.Bin2Float()
        pop_son_F.Bin2Float()

        # Evaluate new raw fitnesses
        pop_son_M.update_Fit(eval_norm_fit=False)
        pop_son_F.update_Fit(eval_norm_fit=False)

        # Update supportive parameters
        self.min_fit_M, self.max_fit_M = pop_son_M.borderFit()
        self.min_fit_F, self.max_fit_F = pop_son_F.borderFit()
        ###
        #print(f'On prev iter:')
        #print(f'min_fit_M={self.min_fit_M}\tmax_fit_M={self.max_fit_M}')
        #print(f'min_fit_F={self.min_fit_F}\tmax_fit_F={self.max_fit_F}\n')
        ###

        if self.class_Ind_M.maximization:
            self.best_fit = max(self.max_fit_M, self.max_fit_F)
        else:
            self.best_fit = min(self.min_fit_M, self.min_fit_F)

        # Get best individ
        best_ind_M = self.pop_M.selBorder(k=1, max_fit=self.class_Ind_M.maximization,
                                          deepcopy_ind=True)[0]
        best_ind_F = self.pop_F.selBorder(k=1, max_fit=self.class_Ind_F.maximization,
                                          deepcopy_ind=True)[0]

        if self.class_Ind_M.maximization:
            self.best_ind = max([best_ind_M, best_ind_F], key=attrgetter('f_fit'))
        else:
            self.best_ind = min([best_ind_M, best_ind_F], key=attrgetter('f_fit'))

        self.class_Ind_M.func_fit_norm = self.class_Ind_M.get_Fit_norm(fit_min=self.min_fit_M, fit_max=self.max_fit_M,
                                                                       maximization=self.class_Ind_M.maximization)
        ###
        #print(f'In Iteration() M:')
        #print(f'self.class_Ind_M={self.class_Ind_M}')
        #print(f'self.class_Ind_M.func_fit_norm={self.class_Ind_M.func_fit_norm}\n')
        ###
        #print(f'self.class_Ind_M.func_fit_norm:\tf(0)={self.class_Ind_M.func_fit_norm(0)}\tf(1)={self.class_Ind_M.func_fit_norm(1)}')
        ###
        self.class_Ind_F.func_fit_norm = self.class_Ind_F.get_Fit_norm(fit_min=self.min_fit_F, fit_max=self.max_fit_F,
                                                                       maximization=self.class_Ind_F.maximization)
        ###
        #print(f'In Iteration() F:')
        #print(f'self.class_Ind_F={self.class_Ind_F}')
        #print(f'self.class_Ind_F.func_fit_norm={self.class_Ind_F.func_fit_norm}\n')
        ###
        ###
        #print(f'self.class_Ind_F.func_fit_norm:\tf(0)={self.class_Ind_F.func_fit_norm(0)}\tf(1)={self.class_Ind_F.func_fit_norm(1)}')
        ###
        #print('-----')
        #print(f'self.class_Ind_M.func_fit_norm={self.class_Ind_M.func_fit_norm}')
        #print(f'self.class_Ind_F.func_fit_norm={self.class_Ind_F.func_fit_norm}')
        #print(f'self.class_Ind_M.func_fit_norm==self.class_Ind_F.func_fit_norm: {self.class_Ind_M.func_fit_norm==self.class_Ind_F.func_fit_norm}')
        #print('-----\n')
        ###
        # Evaluate normed fitnesses of son individs
        pop_son_M.update_Fit(eval_norm_fit=True)
        pop_son_F.update_Fit(eval_norm_fit=True)

        # Forming new elite individs
        l_timing.append(time.perf_counter())  # Timing (->5->)
        pop_son_M.elite_individs = pop_son_M.selBorder(k=len(pop_son_M.elite_individs),
                                                       max_fit=self.class_Ind_M.maximization,
                                                       deepcopy_ind=True)
        pop_son_F.elite_individs = pop_son_F.selBorder(k=len(pop_son_F.elite_individs),
                                                       max_fit=self.class_Ind_F.maximization,
                                                       deepcopy_ind=True)

        self.pop_M = pop_son_M
        self.pop_F = pop_son_F

        l_timing.append(time.perf_counter())  # Timing (->6)

        l_timing_interv = [(l_timing[i] - l_timing[i - 1]) for i in
                           range(1, len(l_timing))]  # Timing intervals evaluating

        return l_timing_interv

    def Necess_log(self, random_state, FileDirectory: str, FileName: str) -> dict:
        """
        Provides basic logging 1 time after evolution is finished.

        :param random_state: Random state from random library before evolution. For repeating
        :param FileDirectory: Path to logging file
        :param FileName: Name of logging file. Should be ended with ".txt"
        """
        dic = {
            # Results of Launch() GA - Convergence epoch + Fit and float chromosomes of solution
            "num_last_epoch": self.num_generation,  # Convergence epoch = num_last_epoch-min_generation_to_stop
            # Convergence epoch - first epoch in final static fitness chain
            "conv_fit": float(self.best_fit),
            "conv_solution": self.best_ind.l_chrom_float,
            # Main characteristics of GA - Name + Task optimization
            "Type GA": self.name,
            # Attributes of individs - Chromosome encoding
            "maximization": self.class_Ind_M.maximization,
            "range_float_chrom": self.class_Ind_M.range_chrom_float,
            "bit_chrom": self.class_Ind_M.bit_chrom,
            # Attributes of population - Population size + Probabilities
            "num_Norm_ind": self.num_ind_M * 2,
            "num_Elite_ind": self.num_elite_ind_M * 2,
            "CXPB_M": self.pop_M.f_cross_prob,
            "CXPB_F": self.pop_F.f_cross_prob,
            "MUTB_M": self.pop_M.f_mut_prob,
            "MUTB_F": self.pop_F.f_mut_prob,
            # Other characteristics of GA - Criteria of convergence + Random.seed
            "delta_fit": self.delta_fit,
            "num_epoch_to_stop": self.num_epoch_to_stop,
            "left_epoch_bound": self.left_epoch_bound,
            "right_epoch_bound": self.right_epoch_bound,
            "random_state": random_state
        }
        JSON_Create(dic, FileDirectory=FileDirectory, FileName=FileName)

        return dic

    def Min_log_create(self, FileDirectory: str = None, FileName: str = None,
                       add_param_names: List[str] = [], add_param_attr: List[str] = []) -> None:
        """
        Creates dir and csv-file for Min_Log dumping data

        :param FileDirectory: Path to logging file
        :param FileName: Name of logging file
        :param add_param_names: Names of additional parameters to log
        :param add_param_attr: Parameters` names to get from population;
        Additional parameters should be attributes of Population object
        :return: None
        """
        param_names = [
                          "num epoch",  # Number of current generation
                          "flag no_fit_change",  # Number of last epochs with static best fitness
                          "best_fit"  # Best fit in population
                      ] + [f"par_{i}" for i in range(len(self.class_Ind_M.range_chrom_float))
                           # Best solution in population
                           ] + add_param_names  # Additional parameters

        self.min_Log_add_param_attr = add_param_attr
        if FileDirectory == None or FileName == None:
            self.min_Log_agent = CSV_Log_agent(param_names, FileDirectory=self.log_FDir,
                                               FileName="min_" + self.log_FName + ".csv")
        else:
            self.min_Log_agent = CSV_Log_agent(param_names, FileDirectory=FileDirectory, FileName=FileName)

    def Min_log_add(self) -> None:
        """
        Provides every-epoch logging on population level.

        :param FileDirectory: Path to logging file
        :param FileName: Name of logging file. Should be ended with ".txt"
        :param add_param_names: Names of additional parameters to log;
        Additional parameters should be attributes of Population object
        :return: None
        """
        # Additional parameteres evaluating
        l_add_param = [getattr(self, par) for par in self.min_Log_add_param_attr]

        l_param = [
                      self.num_generation,  # Number of current generation
                      self.epoch_static_fit,  # Number of epochs with static fit
                      self.best_fit  # Best fit in population
                  ] + [par for par in self.best_ind.l_chrom_float  # Best solution in population
                       ] + l_add_param  # Additional parameteres

        self.min_Log_agent.add_sol(l_param=l_param)

    def Max_log_create(self, FileDirectory: str = None, FileName: str = None,
                       add_param_names: List[str] = [], add_param_attr: List[str] = []) -> None:
        """
        Creates dir and csv-file for Max_Log dumping data

        :param FileDirectory: Path to logging file
        :param FileName: Name of logging file
        :param add_param_names: Names of additional parameters to log
        :param add_param_attr: Parameters` names to get from population;
        Additional parameters should be attributes of Individ object
        :return: None
        """
        param_names = [
                          "num epoch",  # Number of current generation
                          "fit",  # Individ`s fit
                          "b_elite"  # Elite or not
                      ] + [f"par_{i}" for i in range(len(self.class_Ind_M.range_chrom_float))  # Parameters of individ
                           ] + ["Gender"] + add_param_names  # Gender + Additional parameters

        self.max_Log_add_param_attr = add_param_attr
        if FileDirectory == None or FileName == None:
            self.max_Log_agent = CSV_Log_agent(param_names, FileDirectory=self.log_FDir,
                                               FileName="max_" + self.log_FName + ".csv")
        else:
            self.max_Log_agent = CSV_Log_agent(param_names, FileDirectory=FileDirectory, FileName=FileName)

    def Max_log_add(self) -> None:
        """
        Provides every-epoch logging on population level.

        :param FileDirectory: Path to logging file
        :param FileName: Name of logging file. Should be ended with ".txt"
        :param add_param_names: Names of additional parameters to log;
        Additional parameters should be attributes of Individ object
        :return: None
        """
        # Passing through elite individs
        # Male individs
        for ind in self.pop_M.elite_individs:
            # Additional parameters evaluating
            l_add_param = [getattr(self, par) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          1  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameters
                           ] + ['M'] + l_add_param  # Gender + Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)

        # Female individs
        for ind in self.pop_F.elite_individs:
            # Additional parameters evaluating
            l_add_param = [getattr(self, par) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          1  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameters
                           ] + ['F'] + l_add_param  # Gender + Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)

        # Passing through normal individs
        # Male individs
        for ind in (self.pop_M.individs + self.pop_F.individs):
            # Additional parameters evaluating
            l_add_param = [getattr(self, par) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          0  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameteres
                           ] + ['M'] + l_add_param  # Gender + Additional parameteres

            self.max_Log_agent.add_sol(l_param=l_param)

        # Female individs
        for ind in (self.pop_M.individs + self.pop_F.individs):
            # Additional parameteres evaluating
            l_add_param = [getattr(self, par) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          0  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameteres
                           ] + ['F'] + l_add_param  # Gender + Additional parameteres

            self.max_Log_agent.add_sol(l_param=l_param)

    def Launch(self):
        """
        Conducts one launch of GA. Use Iteration() and Preiteration().
        Contains stop criteria of GA.
        """
        # Remembering random state of random library (for possobility of repeating an experiment)
        random_state = random.getstate()
        random.setstate(random_state)

        # PREITERATION - 0 epoch
        self.Preiteration()

        self.num_generation = 0
        self.epoch_static_fit = 0
        # fit_dyn = [self.best_fit]

        # -== Logging preprocessing ==-
        if self.Min_logging:
            self.Min_log_create()
        if self.Max_logging:
            self.Max_log_create()  # add_param_names: List[str] = [], add_param_attr: List[str] = []
        # --====--

        # Main evolution
        l_timing_interv_sum = [0] * len(self.__class__.l_timing_names)  # For time remembering

        # Condition for continuing evolution
        '''
        while (((self.epoch_static_fit < self.num_epoch_to_stop) or (self.num_generation < self.min_generation_to_stop))
        and (self.num_generation < self.max_generation_to_stop)):
        '''
        while ((self.num_generation < self.left_epoch_bound)
        or ((self.num_generation < self.right_epoch_bound) and (self.epoch_static_fit < self.num_epoch_to_stop))):

            prev_best_fit = self.best_fit  # Remember previous best fitness for processing stop criteria

            # ITERATION
            #print('---')
            l_timing_interv = self.Iteration()
            self.num_generation += 1

            # Verifying dynamic of best fitness
            # Count number of static epochs
            if abs(self.best_fit - prev_best_fit) <= (
                    max(self.max_fit_M, self.max_fit_F) - min(self.min_fit_M, self.min_fit_F)) * self.delta_fit:
                self.epoch_static_fit += 1
            else:
                self.epoch_static_fit = 0

            # Summarising timing
            for i in range(len(l_timing_interv_sum)):
                l_timing_interv_sum[i] += l_timing_interv[i]

            # -== Min/Max Logging ==-
            if self.Min_logging:
                self.Min_log_add()
            if self.Max_logging:
                self.Max_log_add()
            # --====--

            # Flowing results and statistics print
            string = (
                f"-- End of {self.num_generation} evolution\t"
                f"best fit = {round(self.best_fit, 4)}\t"
                f"ep. with stc fit: {self.epoch_static_fit}/{self.num_epoch_to_stop}\t"
                f"Sum TIME: {'{:0.2e}'.format(sum(l_timing_interv_sum))}"
                " --")
            #print(string, end='\r')
            print(string, end='\n')

            # fit_dyn.append(deepcopy(self.best_fit))

        print()
        string = "TIME [sec]: "
        for name, time in zip(self.__class__.l_timing_names, l_timing_interv_sum):
            string += f"{name}: {'{:0.2e}'.format(time)}\t"
        print(string, end='\n')

        # Necessary logging
        dic = self.Necess_log(random_state=random_state, FileDirectory=self.log_FDir,
                              FileName="ncss_" + self.log_FName + ".txt")
        print(dic)

        # print(fit_dyn)

###Задачи:
# 3. Решение модельной задачи
# 4. Создание инфографики по структуре классов
#
# 5. Создать модификации GA + з. Коммивояжера/рюкзак
# 6. Создать модификации GGA
