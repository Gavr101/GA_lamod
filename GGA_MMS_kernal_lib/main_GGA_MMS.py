from GGA_MS_kernal_lib.Pop_GGA_MS import Population_GGA_MS_M, Population_GGA_MS_F
from GGA_MS_kernal_lib.GGA_MS import class_GGA_MS


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    GGA = class_GGA_MS(class_Pop_M=Population_GGA_MS_M,
                    class_Pop_F=Population_GGA_MS_F,
                    num_ind=10, num_elite_ind=2,
                    f_mut_prob_M=0.05,
                    f_mut_prob_F=0.005,
                    num_epoch_to_stop=100, delta_fit=0.05,
                    log_FDir='Log_try_GGA', log_FName='1',
                    left_epoch_bound=0,
                    f_cross_prob=1,
                    Min_logging=True, Max_logging=True)

    GGA.Launch()
