#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:57:15 2021

@author: martinkeller-ressel
"""


import sympy as sp
import sympy.polys.monomials as sp_mono
import sympy.polys.orderings as sp_order
import sympy.polys.polytools as sp_poly
import sympy.matrices as sp_matrices
import sys

# Todo proper imports, affine 

class MarkovGenerator:
    """
    This class represents the infinitesimal Generator of a continuous Markov process.
    It provides methods related to the `polynomial property' of certain Markov generators; see Cuchiero, Keller-Ressel and Teichmann (2002).
    In particular, it allows to symbolically calculate moments of continuous-time Markov processes with the 'polynomial proerty'.
    """
    
    def __init__(self, name="", formula=None, stateVariables=None, function="f"):
        
        """
        Constructor of the MarkovGenerator class.
        
        Markov Generators can be constructed either by providing an explicit ``formula`` as a Sympy expression or
        by providing a ``name`` from a dictionary of predefined Markov Generators.
        If the generator is constructed from a formula, the a
        
        
        Parameters
        ----------
        formula: sympy.core.expr.Expr
            A sympy expression representing the Markov generator.
            
        stateVariables: list[sympy.core.symbol.Symbol] or list[str]
            The state variables of the Markov generator
            
        function: sympy.core.symbol.FunctionClass or str
            The function argument of the Markov generator.
            
        name: str
            The name of a predefined Markov generator.
            
        Returns
        -------
        A MarkovGenerator object.
        
        """
        if formula is None:
             [formula, stateVariables, function] = _predefinedGenerators(name)
        self.formula = formula if isinstance(formula,sp.Expr) else sp.sympify(formula)
        self.stateVariables = [var if isinstance(var,sp.Symbol) else sp.Symbol(var) for var in stateVariables]
        self.function = function if isinstance(function,sp.FunctionClass) else sp.Function(function)

    def apply(self, expression):
        """
        Applies the Markov generator to a Sympy expression
        
        The function argument of the Markov generator is substituted by 'expression' and evaluated; the result is returned. 
        Note: The expressions should be functions of the state variables of the Markov Generator

        Parameters
        ----------
        expression : sympy.core.expr.Expr
            A sympy expression to which the Markov generator can be applied. 
          
        Returns
        -------
        sympy.core.expr.Expr
            The result of applying the Markov generator to ``expression``.

        """
        replacement = sp.Lambda(tuple(self.stateVariables),expression)
        newExpression = self.formula.replace(self.function,replacement)
        return(newExpression.doit())

    def applyList(self, expressions):
        """
        Applies the Markov generator to a list of Sympy expressions
        
        The function argument of the Markov generator is substituted by each element of 
        'expressions' and evaluated; the results are returned as a list. 
        Note: The expressions should be functions of the state variables of the Markov Generator

        Parameters
        ----------
        expressions : list[sympy.core.expr.Expr]
            A list of sympy expressions to which the Markov generator can be applied. 
          
            
        Returns
        -------
        list[sympy.core.expr.Expr]
            The result of applying the Markov generator to each element of ``expressions``.

        """
        evaluatedExpressions = list()
        for expression in expressions:
            replacement = sp.Lambda(tuple(self.stateVariables),expression)
            newExpression = self.formula.replace(self.function,replacement)
            evaluatedExpressions.append(newExpression.doit())
        return(evaluatedExpressions)

    def polyBasis(self,deg=2,order='grevlex',max_count=100):
        """
        Returns a basis of polynomials in the state Variables 
        of the Markov generator. 
        
        Parameters
        ----------
        deg: int
            The maximal degree of polynomials included in the basis; 
            controls the size of the basis
            
        order: str
            A string as accepted by sympy.polys.orderings.monomial_key.
            Determines the order of basis elements. Default is 'grevlex' corresponding to
            graded reverse lexicographic order.
            
        max_count: int
            The maximal number of basis elements returned.
            The size of the basis can grow quickly with degree and the number of state Variables. Before the basis is calculated, 
            the number of basis elements is calculated. If this number is greater than `max_count` a warning is printed and None is returned.
            
          
        Returns
        -------
        list[sympy.core.expr.Expr] or None
            A basis of polynomials.
        
        """
        count = sp_mono.monomial_count(len(self.stateVariables), deg)
        if count > max_count:
            print('Basis size exceeds max_count, returning None')
            return None
        else:
            basis = sorted(sp_mono.itermonomials(self.stateVariables, deg), key=sp_order.monomial_key(order, self.stateVariables[::-1]))
            return(basis)
        
    def hasPolyProperty(self,deg=5,max_count=100):
        """
        Checks whether the Markov generator has the `polynomial property'.
        
        A Markov generator has the polynomial property, if it maps any given polynomial to another polynomial of
        less or equal degree. See Cuchiero, Keller-Ressel and Teichmann (2002) for details. 
        This function checks the polynomial property up to the maximal degree ``deg``.

        Parameters
        ----------
        deg : int
            The degree up to which the polynomial property of the generator is checked.
            
        max_count : int
            The maximal number of checks performed.
            The number of checks can grow quickly with degree and the number of state variables. Before the polynomial property is checked,
            the number of checks that must be performed is calculated. If this number is greater than `max_count` a warning is printed and None is returned.

        Returns
        -------
        bool or None
            The result of checking for the polynomial property.

        """
        count = sp_mono.monomial_count(len(self.stateVariables), deg)
        if count > max_count:
            print('Number of conditions to check exceeds max_count, returning None')
            return None
        else:
            basis = sorted(sp_mono.itermonomials(self.stateVariables, deg), key=sp_order.monomial_key('grevlex', self.stateVariables[::-1]))
            transformation = self.applyList(basis)
            polyCheck = [sp.sympify(expr).is_polynomial(*self.stateVariables) for expr in transformation]
            if not all(polyCheck): return False
            (basisAsPoly,info) = sp_order.monomial_key(basis,self.stateVariables)
            (transformationAsPoly,info) = sp_poly.parallel_poly_from_expr(transformation,self.stateVariables)
            deg_in = [p.total_degree() for p in basisAsPoly]
            deg_out = [p.total_degree() for p in transformationAsPoly]
            degreeCheck = [deg_out[i] <= deg_out[i] for i in range(len(deg_in))]
            return(all(degreeCheck))
            
    def polyMatrix(self,deg=2,order='grevlex',max_count=100,basis=None):
        """
        Calculates the polynomial generator matrix of the Markov generator.
        
        If the Markov generator has the polynomial property, then its action on a basis 
        of polynomials can be expressed as a matrix ``A``. If a basis is given, then this basis is used
        to calculate ``A``. If no basis is given, then a basis is generated based on the parameters ``deg``and ``order``, see also
        polyBasis. 
     
        Parameters
        ----------
        deg : int
            The maximal degree of polynomials included in the basis for which the polynomial generator Matrix is calculated.
        order : str
            A string as accepted by sympy.polys.orderings.monomial_key.
            Determines the order of basis elements. Default is 'grevlex' corresponding to
            graded reverse lexicographic order.
        max_count: int
            The maximal basis size used, and the maximal dimension of the polynomial generator matrix returned. 
            The size of the basis can grow quickly with degree and the number of state variables. Before the basis and the matrix ``A``are calculated, 
            the number of basis elements is calculated. If this number is greater than `max_count` a warning is printed and None is returned.
        basis : list[sympy.core.expr.Expr] or None
            A basis of polynomials, given as a list of Sympy expressions. Overrides the parameters ``deg``, ``order`` and ``max_count``, if not None.

        Returns
        -------
        basis : list[sympy.core.expr.Expr]
            The basis used to calculate the polynomial generator matrix.
        A : sympy.matrices.Matrix
            The polynomial generator matrix, represented by a Sympy Matrix.
        """
        
        if basis is None:
            count = sp_mono.monomial_count(len(self.stateVariables), deg)
            if count > max_count:
                print('Number of conditions to check exceeds max_count, returning None')
                return None
            else:
                basis = sorted(sp_mono.itermonomials(self.stateVariables, deg), key=sp_order.monomial_key(order, self.stateVariables[::-1]))
        transformation = self.applyList(basis)
        (basisAsPoly,info) = sp_poly.parallel_poly_from_expr(basis,self.stateVariables)
        (transformationAsPoly,info) = sp_poly.parallel_poly_from_expr(transformation,self.stateVariables)
        deg_in = [p.total_degree() for p in basisAsPoly]
        deg_out = [p.total_degree() for p in transformationAsPoly]
        degreeCheck = [deg_out[i] <= deg_out[i] for i in range(len(deg_in))]
        if not all(degreeCheck): 
            print('Generator does not have the polynomial property. Returning None')
            return None
        A = sp.Matrix([[p.coeff_monomial(b) for p in transformationAsPoly] for b in basis])
        A.simplify()
        diff = sp.Matrix(basis).transpose() * A - sp.Matrix(transformation).transpose()
        if(diff.norm() == 0): 
            return (basis,A)
        else:
            print('Markov generator does not preserve the given basis. Returning None')
            return None
     
    def moment(self,which_moment,time=sp.Dummy("t"),showProgress=True, max_count=50):
        """
        Calculate moments of a (polynomial) stochastic process represented by its Markov generator.
        
        This function uses symbolic matrix exponentiation to calculate the exact moments of a stochastic
        process with the polynomial property; a method introduced in Cuchiero, Keller-Ressel, Teichmann (2002).
        This method can be quite time-consuming.
        
        Parameters
        ----------
        which_moment : list[sympy.core.expr.Expr or tuple]
            A list specifying the (mixed) moments to be caluated either as polynomials (e.g.x**3 or x**2*y) in the generator's state variables,
            or as tuples (e.g. (3,0) or (2,1)) representing the desired degrees of the state variables.  
        time : sympy.core.symbol.Symbol, optional
            The symbol that should be used as time variable in the moment calculation. The default is sp.Dummy("t").
        showProgress : bool, optional
            Should a progress bar be shown? Symbolic matrix exponentiation can be time consuming. 
            Therefore a progress bar may have a reassuring effect on the user. The default is True.
        max_count : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """

        # Todo: Accommodate for initial conditions
        moment_list = self._sanitize_moment_spec(which_moment,max_count) 
        (momentsAsPoly, info) = sp_poly.parallel_poly_from_expr(moment_list,self.stateVariables)
        deg_list = [p.total_degree() for p in momentsAsPoly]
        max_degree = max(deg_list)
        (basis,A) = self.polyMatrix(deg=max_degree,max_count=max_count)
        MList = []
        for index,m in enumerate(moment_list):
            representation = [momentsAsPoly[index].coeff_monomial(b) for b in basis]
            MList.append(representation)
        M = sp.Matrix(MList)
        out = M*sp.exp(A.transpose()*time)*sp.Matrix(basis)
        # Implememnt using A.eigenvects()
        out.simplify()
        return(out)
    
    def stationaryMoment(self,which_moment,max_count=100):
        pass

    def _sanitize_moment_spec(self,which_moment,max_count=100):
        moment_list_raw = which_moment if isinstance(which_moment,list) else [which_moment]
        moment_list = []
        for m in moment_list_raw:
            if(isinstance(m,sp.Expr)):
                mAsExpr = m
            elif(isinstance(m,int) and len(self.stateVariables) == 1):
                mAsExpr = sp.Pow(self.stateVariables[0],m)
            elif(isinstance(m,tuple)):
                z = zip(self.stateVariables,m)
                powers = [sp.Pow(base,exponent) for (base,exponent) in z]
                mAsExpr = sp.Mul(*powers)
            else:
                print("Skipping element " + m + " of which_moment; cannot understand format")
                next
            moment_list.append(mAsExpr)
        return moment_list
    
def putzerAlgorithm(A,time = sp.Dummy("t"), rightFactor = None, simplifyAlways=True, showProgress=True):
    """
    Implements Putzer's Algorithm for the symbolic calculation of matrix exponentials
    
    See https://en.wikipedia.org/wiki/Matrix_differential_equation#Putzer_Algorithm_for_computing_eAt for a description of the algorithm

    Parameters
    ----------
    A : sympy.matrices.Matrix
        A square matrix, of which the exponential exp(A*t) should be calulated
    time : sympy.core.symbol.Symbol
        The time variable. The default is sympy.core.symbol.Dummy("t")
    rightFactor : sympy.matrices.Matrix
        If not None then exp(A*t)*rightFactor is returned instead of exp(A*t). The number of rows of rightFactor should be equal to the dimension of A. 
        If the number of columns of rightFactor is small, then passing rightFactor as an argument is more efficient than calculating exp(A*t) first and then multiplying with rightFactor.
        The default is None.
    simplifyAlways : bool, optional
        Should expressions be simplified in every iteration of Putzer's algorithm (or only at the end)? Simplifying in every step seems to be more efficient in most cases.
        The default is True.
    showProgress : bool, optional
        Should a progress bar be shown? For large matrices (more than ten rowns/colums) symbolic exponentiation is time consuming. 
        Therefore a progress bar may have a reassuring effect on the user. The default is True.

    Returns
    -------
    AExp : sympy.matrices.Matrix
        The matrix exp(A*t) (if rightFactor is None) or the matrix exp(A*t)*rightFactor (is rightFactor is not None).
        If time is set to a non-default value, this sybol is used in place of t.

    """
    eigenvalues = A.eigenvals(multiple=True)
    n = len(eigenvalues)
    P0 = rightFactor if rightFactor is not None else sp_matrices.eye(n)
    r0 = sp.exp(eigenvalues[0]*time)
    PList = [P0]
    rList = [r0]
    AExp = P0 * r0
    for i in range(n-1):
        P =  A*PList[-1] - eigenvalues[i]*PList[-1]
        if simplifyAlways : P.simplify()
        PList.append(P)
        integrand = sp.exp(-eigenvalues[i+1]*time)*rList[-1]
        r = sp.exp(eigenvalues[i+1]*time)*sp.integrate(integrand,(time,0,time))
        if simplifyAlways : r.simplify()
        rList.append(r)
        AExp = AExp + P * r
        if showProgress: progress(i+1, n-1, status='Computing Matrix Exponential')

    if not simplifyAlways: AExp.simplify()
    return AExp

    ## use assumptions!

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    #sys.stdout.flush()
    
      
  
def _BM_generator(**kwargs):
    try:
        dimension = kwargs['d']
    except KeyError:
        dimension = 1
    if(dimension == 1):
        return ["1/2 * f(x).diff(x,2)",["x"],"f"]
    elif(dimension == 2):
        return ["1/2 * f(x,y).diff(x,2,y,2)",["x","y"],"f"]
    elif(dimension == 3):
        return ["1/2 * f(x,y,z).diff(x,2,y,2,z,2)",["x","y","z"],"f"]
    else:
        stateVariables = ["x" + str(i+1) for i in range(dimension)]
        
        return None
    
    

def _OU_generator(**kwargs):
    pass
    
def _Heston1_generator(**kwargs):
    return ["-1/2*v*f(x,v).diff(x,1) + 1/2*v*f(x,v).diff(x,2) - lamda*(v - theta)*f(x,v).diff(v,1) + 1/2*eta**2*v*f(x,v).diff(v,2) + rho*eta*v*f(x,v).diff(x,v)", ["x", "v"], "f"]
    
def _Heston2_generator(**kwargs):
    x, theta, rho = sp.symbols("x theta rho")
    v = sp.symbols("v", nonnegative = True)
    lamda, eta = sp.symbols("lamda, eta", positive=True)
    f = sp.Function("f")
    return [sp.UnevaluatedExpr(-sp.Rational(1,2)*v*f(x,v).diff(x,1) + sp.Rational(1,2)*v*f(x,v).diff(x,2) - lamda*(v - theta)*f(x,v).diff(v,1) + sp.Rational(1,2)*eta**2*v*f(x,v).diff(v,2) + rho*eta*v*f(x,v).diff(x,v)), [x, v], f]
    

def _predefinedGenerators(name):
    if(name=="BM"): return _BM_generator(1)
    if(name=="Heston1"): return _Heston1_generator()
    if(name=="Heston2"): return _Heston2_generator()
    
#h = MarkovGenerator(name="Heston")
#h.moment((0,2))