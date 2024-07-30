from typing import List

from GA_kernal_lib.GA_kernal import class_NGA
from GA_kernal_lib.Logging_kernal import CSV_Log_agent, JSON_Create


class class_NGA_1hc(class_NGA):

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
            "conv_solution": self.best_ind.l_chrom_bin[0],
            # Main characteristics of GA - Name + Task optimization
            "Type GA": self.name,
            # Attributes of individs - Chromosome encoding
            "maximization": self.class_Ind.maximization,
            "bit_chrom": self.class_Ind.bit_chrom,
            # Attributes of population - Population size + Probabilities
            "num_Norm_ind": len(self.pop.individs),
            "num_Elite_ind": len(self.pop.elite_individs),
            "CXPB": self.pop.f_cross_prob,
            "MUTB": self.pop.f_mut_prob,
            # Other characteristics of GA - Criteria of convergence + Random.seed
            "delta_fit": self.delta_fit,
            "num_epoch_to_stop": self.num_epoch_to_stop,
            "min_generation_to_stop": self.min_generation_to_stop,
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
            ] + [f"par_{i}" for i in range(len(self.class_Ind.bit_chrom))  # Best solution in population
            ] + add_param_names  # Additional parameters


        self.min_Log_add_param_attr = add_param_attr
        if FileDirectory==None or FileName==None:
            self.min_Log_agent = CSV_Log_agent(param_names, FileDirectory=self.log_FDir,
                                               FileName="min_"+self.log_FName+".csv")
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
            ] + [par for par in self.best_ind.l_chrom_bin  # Best solution in population
            ] + l_add_param  # Additional parameters


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
            ] + [f"par_{i}" for i in range(len(self.class_Ind.bit_chrom))  # Best solution in population
            ] + add_param_names  # Additional parameters


        self.max_Log_add_param_attr = add_param_attr
        if FileDirectory==None or FileName==None:
            self.max_Log_agent = CSV_Log_agent(param_names, FileDirectory=self.log_FDir,
                                               FileName="max_"+self.log_FName+".csv")
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
                ] + [par for par in ind.l_chrom_bin  # Individ`s parameters
                ] + l_add_param  # Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)


        # Passing through elite individs
        for ind in self.pop.individs:
            # Additional parameters evaluating
            l_add_param = [getattr(self, ind) for par in self.max_Log_add_param_attr]

            l_param = [
                self.num_generation,  # Number of generation
                ind.f_fit,  # Individ`s fit
                0  # b_Elite
                ] + [par for par in ind.l_chrom_bin  # Individ`s parameters
                ] + l_add_param  # Additional parameters

            self.max_Log_agent.add_sol(l_param=l_param)