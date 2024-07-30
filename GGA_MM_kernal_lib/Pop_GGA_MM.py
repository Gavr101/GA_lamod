from GGA_MM_kernal_lib.Ind_GGA_MM import Individ_GGA_MM_M, Individ_GGA_MM_F
from GGA_kernal_lib.Pop_GGA import Population_GGA_M, Population_GGA_F


class Population_GGA_MM_M(Population_GGA_M):
    """Male population"""
    class_Ind = Individ_GGA_MM_M


class Population_GGA_MM_F(Population_GGA_F):
    """Female population"""
    class_Ind = Individ_GGA_MM_F

