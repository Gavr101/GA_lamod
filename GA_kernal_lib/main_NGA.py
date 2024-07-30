from Pop_kernal import Population
from GA_kernal import class_NGA

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    NGA = class_NGA(class_Pop=Population,
                    num_ind=5,
                    num_elite_ind=1,
                    f_mut_prob=0.01,
                    num_epoch_to_stop=100,
                    delta_fit=0.05,
                    log_FDir='Log_try_NGA',
                    log_FName='1',
                    Min_logging=True,
                    Max_logging=True)

    NGA.Launch()