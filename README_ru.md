# Общее описание 
В данном проекте реализован генетический алгоритм с бинарным кодированием, а также три его гендерных модификации, описанных в статье "Study of Modifications of Gender Genetic Algorithm" (http://dx.doi.org/10.1007/978-3-031-44865-2_30).

В директории <GA_1hc> реализован игрушечный пример с решением задачи 1-hot-coding.

В директории <Ackley_model_task> реализовано решение задачи оптимизации функции Ackley 4-мя типами генетического алгоритма.
---
# Генетические алгоритмы
Реализация не использует сторонних пакетов. 
Иерархия директорий: 

0) Базовые директории: <GA_kernal_lib> &#8594; <GGA_kernal_lib>;
1) Гендерные модификации: <GGA_MS_kernal_lib>, <GGA_MM_kernal_lib>, <GGA_MMS_kernal_lib>;
2) Конкретизация решаемой оптимизационной задачи. Реализуется на последнем этапе. Примеры: <GA_1hc>, <Ackley_model_task>.
---

*Программный код реализован при поддержке гранта Российского научного фонда № 22-12-00138, https://rscf.ru/project/22-12-00138/ ; стипендии развития теоретической физики и математики фонда «БАЗИС» №23-2-1-65-1.*