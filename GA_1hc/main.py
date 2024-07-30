from GA_1hc import class_NGA_1hc
from Pop_1hc import Population_1hc

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    NGA_1hc = class_NGA_1hc(class_Pop=Population_1hc,
                            num_ind=5,
                            num_elite_ind=1,
                            f_mut_prob=0.01,
                            num_epoch_to_stop=100,
                            delta_fit=0.05,
                            log_FDir='Log_try',
                            log_FName='1',
                            Min_logging=True,
                            Max_logging=True)

    NGA_1hc.Launch()
