import collections as co
import functools as ft

import sympy as sy

from .types import SymbolTable, FunctionTable, PyTree
from .default import DEFAULT_FUNCTION_TABLE
from .core import handler, translate


__all__ = [
    "to_jax"
]
    

def to_jax(
    expr: sy.Expr, 
    *, 
    symbol_table: SymbolTable = {}, 
    extra_function_table: FunctionTable = {},
) -> PyTree:
    '''Transpiles sympy expressions into JAX graphs.
    
       Takes a sympy expression and a symbol table containing numeric values and executes them as a JAX graph.
       This works by symbol substitution and evaluation according to the symbols provided in the symbol table.
       
       Parameters
       ----------
       expr: sy.Expr
             The sympy expression that is to be transpiled.
       symbol_table: sy2jax.SymbolTable
             The symbol table used for symbol substitution.
       extra_function_table: sy2jax.FunctionTable
             A function table of extra function substitutions.
             Usually can be left blank. Only required for customization.
             
       Returns
       -------
       sys2jax.PyTree
           The result of the running the corresponding JAX graph.       
    '''        
    function_table = co.ChainMap(extra_function_table, DEFAULT_FUNCTION_TABLE)
    return translate(expr, symbol_table=symbol_table, function_table=function_table)