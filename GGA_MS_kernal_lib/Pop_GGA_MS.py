from typing import List

from GGA_MS_kernal_lib.Ind_GGA_MS import Individ_GGA_MS_M, Individ_GGA_MS_F
from GGA_kernal_lib.Pop_GGA import Population_GGA_M, Population_GGA_F


class Population_GGA_MS_M(Population_GGA_M):
    """Male population"""
    class_Ind = Individ_GGA_MS_M

    def Select(self, k=1, deepcopy_ind=True) -> List[class_Ind]:
        return super()._selTournament(tournsize=3, k=k, deepcopy_ind=deepcopy_ind)


class Population_GGA_MS_F(Population_GGA_F):
    """Female population"""
    class_Ind = Individ_GGA_MS_F

    def Select(self, k=1, deepcopy_ind=True) -> List[class_Ind]:
        return super()._selRoulette(k=k, deepcopy_ind=deepcopy_ind)