from typing import List
from GA_kernal_lib.Ind_kernal import Individ


class _Individ_GGA_MMS(Individ):
    """Pad class for gendered individs. Made to determine cross and mutation methods."""

    @staticmethod
    def cross(ind1, ind2, self_change=True) -> List:
        return ind1.__class__._cross_1p(ind1, ind2, self_change=self_change)



class Individ_GGA_MMS_M(_Individ_GGA_MMS):
    """Male individ. Made to separate Male`s and Female`s name spaces. Helps avoid problem with norming function."""

    def mutate(self, self_change=True) -> List:
        return super().mutate_1b_LefTri(self_change=self_change)


class Individ_GGA_MMS_F(_Individ_GGA_MMS):
    """Female individ. Made to separate Male`s and Female`s name spaces. Helps avoid problem with norming function."""

    def mutate(self, self_change=True) -> List:
        return super().mutate_1b_RightTri(self_change=self_change)