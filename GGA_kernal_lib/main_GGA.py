from GGA_kernal_lib.Pop_GGA import Population_GGA_M, Population_GGA_F
from GGA_kernal_lib.GGA_kernal import class_GGA

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    GGA = class_GGA(class_Pop_M=Population_GGA_M,
                    class_Pop_F=Population_GGA_F,
                    num_ind=10, num_elite_ind=2,
                    f_mut_prob_M=0.05,
                    f_mut_prob_F=0.005,
                    num_epoch_to_stop=100, delta_fit=0.05,
                    log_FDir='Log_try_GGA', log_FName='1',
                    min_generation_to_stop=0,
                    f_cross_prob=1,
                    Min_logging=True, Max_logging=True)

    GGA.Launch()
