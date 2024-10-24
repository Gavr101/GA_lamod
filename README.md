# General Description
This project implements a genetic algorithm with binary encoding, as well as three of its gender modifications described in the article "Study of Modifications of Gender Genetic Algorithm" (http://dx.doi.org/10.1007/978-3-031-44865-2_30).

Results of their application to model problems:
![](/pictures/picture_1.png)
![](/pictures/picture_2.png)

In the directory <GA_1hc>, a toy example is implemented for solving the 1-hot-coding problem.

In the directory <Ackley_model_task>, a solution to the optimization problem of the Ackley function is implemented using four types of genetic algorithms.

---
# Genetic Algorithms
The implementation does not use third-party packages.
Directory hierarchy:

0) Base directories: <GA_kernal_lib> → <GGA_kernal_lib>;
1) Gender modifications: <GGA_MS_kernal_lib>, <GGA_MM_kernal_lib>, <GGA_MMS_kernal_lib>;
2) Specification of the optimization problem being solved. Implemented at the final stage. Examples: <GA_1hc>, <Ackley_model_task>.
---

*The software code was developed with the support of the Russian Science Foundation grant No. 22-12-00138, https://rscf.ru/project/22-12-00138/; the scholarship for the development of theoretical physics and mathematics from the "BAZIS" foundation No. 23-2-1-65-1.*