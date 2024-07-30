from typing import List

from Ind_1hc import Individ_1hc
from GA_kernal_lib.Pop_kernal import Population


class Population_1hc(Population):
    class_Ind = Individ_1hc

    def Select(self, k=1, deepcopy_ind=True) -> List[Individ_1hc]:
        return super()._selRoulette(k=k, deepcopy_ind=deepcopy_ind)

    def Bin2Float(self) -> None:
        pass

    def Float2Bin(self) -> None:
        pass
