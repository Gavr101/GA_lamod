from GGA_kernal_lib.Ind_GGA import Individ_GGA_M, Individ_GGA_F
from GA_kernal_lib.Pop_kernal import Population


class Population_GGA_M(Population):
    """Male population"""
    class_Ind = Individ_GGA_M


class Population_GGA_F(Population):
    """Female population"""
    class_Ind = Individ_GGA_F

