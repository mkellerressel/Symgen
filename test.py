#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:56:58 2021

@author: martinkeller-ressel
"""

from timeit import default_timer as timer
import generator as gen
from sympy import symbols
#from sympy.solvers.ode.systems import linodesolve, matrix_exp

h = gen.MarkovGenerator("Heston2")

start = timer()
(basis,A) = h.polyMatrix(3)
end = timer()
print(end-start)

# start = timer()
# P, J = A.jordan_form()
# #P.inv()
# end = timer()h = gen.MarkovGenerator("Heston2")
# print(end-start)

# start = timer()
# sol = linodesolve(A,symbols("t"),type="type1")
# #P.inv()
# end = timer()
# print(end-start)

start = timer()
sol = gen.putzerAlgorithm(A,symbols("t"),None,True)
end = timer()
print(end-start)

# start = timer()
# sol = putzerAlgorithm(A,symbols("t"),False)
# end = timer()
# print(end-start)