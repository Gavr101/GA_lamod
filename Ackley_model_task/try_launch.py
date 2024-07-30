from NGA_ackley32 import Population_NGA, Individ_NGA
from GA_kernal_lib.GA_kernal import class_NGA

NGA = class_NGA(class_Pop=Population_NGA,
                num_ind=94,
                num_elite_ind=6,
                f_mut_prob=0.01,
                num_epoch_to_stop=200,
                delta_fit=0.001,
                log_FDir='LOG_try',
                log_FName='1',
                Min_logging=True,
                Max_logging=True)

NGA.Launch()
