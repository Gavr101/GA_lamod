from copy import deepcopy
import time
import random
from typing import List

from GA_kernal_lib.Logging_kernal import CSV_Log_agent, JSON_Create


class class_NGA:
    """
    Canonic GA.
    Determines process of evolution, 3-stage logging
    """
    name = "NGA"
    l_timing_names = ['Copy elites', 'SEL', 'CROSS', 'MUT', 'Fit eval', 'Choose elites']

    def __init__(self, class_Pop,
                 num_ind: int, num_elite_ind: int,
                 f_mut_prob: float,
                 num_epoch_to_stop: int, delta_fit: float,
                 log_FDir: str, log_FName: str,
                 left_epoch_bound=0, right_epoch_bound=float("inf"),
                 f_cross_prob=1,
                 Min_logging=True, Max_logging=False, print_pop=False) \
            -> None:
        self.class_Pop = class_Pop
        self.class_Ind = class_Pop.class_Ind

        # Input main characteristics of GA
        self.num_ind = num_ind
        self.num_elite_ind = num_elite_ind
        self.f_mut_prob = f_mut_prob
        self.num_epoch_to_stop = num_epoch_to_stop
        self.delta_fit = delta_fit
        self.left_epoch_bound = left_epoch_bound
        self.right_epoch_bound = right_epoch_bound
        self.f_cross_prob = f_cross_prob

        # Input logging characteristics of GA
        self.log_FDir = log_FDir
        self.log_FName = log_FName
        self.Min_logging = Min_logging
        self.Max_logging = Max_logging
        self.print_pop = print_pop

        # Supportive parameters
        self.max_fit = None
        self.min_fit = None
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
        self.pop = self.class_Pop().Random_generation(num_ind=self.num_ind,
                                                      num_elite_ind=self.num_elite_ind,
                                                      f_mut_prob=self.f_mut_prob,
                                                      f_cross_prob=self.f_cross_prob)
        self.pop.Bin2Float()

        # Update supportive parameters
        self.min_fit, self.max_fit = self.pop.borderFit()

        if self.class_Ind.maximization:
            self.best_fit = self.max_fit
        else:
            self.best_fit = self.min_fit

        # Update norm Fit and remember best fit
        self.class_Ind.func_fit_norm = self.class_Ind.get_Fit_norm(fit_min=self.min_fit,
                                                                   fit_max=self.max_fit,
                                                                   maximization=self.class_Ind.maximization)
        self.pop.update_Fit(eval_norm_fit=True)
        print('\n\n\n----\nInit')

        if self.print_pop: self.pop.my_print()  # Printing all individs characteristics

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

        if self.print_pop: self.pop.my_print()  # Printing all individs characteristics

        # Preliminarily forming son_pop for coping elite individs
        l_timing.append(time.perf_counter())  # Timing (0->)

        pop_son = self.class_Pop(l_elite_ind=deepcopy(self.pop.elite_individs),
                                 f_mut_prob=self.pop.f_mut_prob,
                                 f_cross_prob=self.pop.f_cross_prob)

        # SELECTION
        l_timing.append(time.perf_counter())  # Timing (->1->)

        l_ind_sel1 = self.pop.Select(len(self.pop.individs) // 2, deepcopy_ind=False)
        l_ind_sel2 = self.pop.Select(len(self.pop.individs) // 2, deepcopy_ind=False)

        # CROSSOVER
        l_timing.append(time.perf_counter())  # Timing (->2->)

        l_ind_son1, l_ind_son2 = self.class_Pop.Cross(l_ind1=l_ind_sel1,
                                                      l_ind2=l_ind_sel2,
                                                      f_cross_prob=self.pop.f_cross_prob,
                                                      self_change=False)

        pop_son.individs = l_ind_son1 + l_ind_son2

        # MUTATION
        l_timing.append(time.perf_counter())  # Timing (->3->)

        pop_son.Mut()

        # Assign pop_son to self.individs!!!
        # self.pop.individs = pop_son

        # Evaluate float_chromosomes from bin_chromosomes using Bin2Float()
        l_timing.append(time.perf_counter())  # Timing (->4)

        pop_son.Bin2Float()

        # Evaluate new raw fitnesses
        pop_son.update_Fit(eval_norm_fit=False)

        # Update supportive parameters
        # Get min/max fit
        self.min_fit, self.max_fit = pop_son.borderFit()

        # Get best fit
        if self.class_Ind.maximization:
            self.best_fit = self.max_fit
        else:
            self.best_fit = self.min_fit

        # Get best individ
        self.best_ind = self.pop.selBorder(k=1, max_fit=self.class_Ind.maximization,
                                           deepcopy_ind=True)[0]

        self.class_Ind.func_fit_norm = self.class_Ind.get_Fit_norm(fit_min=self.min_fit, fit_max=self.max_fit,
                                                                   maximization=self.class_Ind.maximization)

        # Evaluate normed fitnesses of son individs
        pop_son.update_Fit(eval_norm_fit=True)

        # Forming new elite individs
        l_timing.append(time.perf_counter())  # Timing (->5->)
        pop_son.elite_individs = pop_son.selBorder(k=len(pop_son.elite_individs),
                                                   max_fit=self.class_Ind.maximization,
                                                   deepcopy_ind=True)
        self.pop = pop_son

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
            "maximization": self.class_Ind.maximization,
            "range_float_chrom": self.class_Ind.range_chrom_float,
            "bit_chrom": self.class_Ind.bit_chrom,
            # Attributes of population - Population size + Probabilities
            "num_Norm_ind": len(self.pop.individs),
            "num_Elite_ind": len(self.pop.elite_individs),
            "CXPB": self.pop.f_cross_prob,
            "MUTB": self.pop.f_mut_prob,
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
                      ] + [f"par_{i}" for i in range(len(self.class_Ind.range_chrom_float))
                           # Best solution in population
                           ] + add_param_names  # Additional parameteres

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
        # Additional parameters evaluating
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
                          "b_elite"  # Best fit in population
                      ] + [f"par_{i}" for i in range(len(self.class_Ind.range_chrom_float))
                           # Best solution in population
                           ] + add_param_names  # Additional parameteres

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
        for ind in self.pop.elite_individs:
            # Additional parameters evaluating
            l_add_param = [getattr(self, ind) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          1  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameters
                           ] + l_add_param  # Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)

        # Passing through normal individs
        for ind in self.pop.individs:
            # Additional parameters evaluating
            l_add_param = [getattr(self, ind) for par in self.max_Log_add_param_attr]

            l_param = [
                          self.num_generation,  # Number of generation
                          ind.f_fit,  # Individ`s fit
                          0  # b_Elite
                      ] + [par for par in ind.l_chrom_float  # Individ`s parameters
                           ] + l_add_param  # Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)

    def Launch(self):
        """
        Conducts one launch of GA. Use Iteration() and Preiteration().
        Contains stop criteria of GA.
        """
        # Remembering random state of random library (for possibility of repeating an experiment)
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
            self.Max_log_create()
        # --====--

        # Main evolution
        l_timing_interv_sum = [0] * len(self.__class__.l_timing_names)  # For time remembering

        # Condition for continuing evolution
        while ((self.num_generation < self.left_epoch_bound)
        or ((self.num_generation < self.right_epoch_bound) and (self.epoch_static_fit < self.num_epoch_to_stop))):
            prev_best_fit = self.best_fit  # Remember previous best fitness for processing stop criteria

            # ITERATION
            #print('---')
            l_timing_interv = self.Iteration()
            self.num_generation += 1

            # Verifying dynamic of the best fitness
            # Count number of static epochs
            if abs(self.best_fit - prev_best_fit) <= (self.max_fit - self.min_fit) * self.delta_fit:
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
        #print(dic)

        # print(fit_dyn)

###Задачи:
# 3. Решение модельной задачи
# 4. Создание инфографики по структуре классов
#
# 5. Создать модификации GA + з. Коммивояжера/рюкзак
# 6. Создать модификации GGA
