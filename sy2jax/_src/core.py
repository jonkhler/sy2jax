import collections as co
import functools as ft
from typing import Callable

import sympy as sy

from .types import (
    PyTree,
    Numeric,
    UnaryOp,
    BinaryOp,
    SympyExprType,
    SymbolTable,
    FunctionTable,
    Translation,
)


__all__ = [
    "type_cast",
    "unary_op",
    "binary_op",
    "call_on_tuple",
    "call",
    "symbol",
    "handler",
]


# TODO: singledispatch is not strictly needed here anymore
# TODO: remove and replace by static cache?
@ft.singledispatch
def handler(
    expr: sy.Expr, *, symbol_table: SymbolTable, function_table: FunctionTable
) -> PyTree:
    """Handles a sympy expression during the transpilation.
    
       Depending on the sympy expression's type, this will dispatch
       a corresponding handler or raise a NotImplementedError if no
       corresponding handler has been registered.
       
       
       Parameters
       ----------
       expr: sy.Expr
             The sympy expression that is to be transpiled.
       symbol_table: sy2jax.SymbolTable
             The symbol table used for symbol substitution.
       function_table: sy2jax.FunctionTable
             The function table of function substitutions.
             
       Returns
       -------
       sys2jax.PyTree
           The result of the running the corresponding JAX graph. 
    """
    raise NotImplementedError(f"Implementation missing for {type(expr)}")


def translate(
    expr: sy.Expr, *, symbol_table: SymbolTable, function_table: FunctionTable,
) -> PyTree:
    function_table = co.ChainMap(function_table, handler.registry)
    for clz in type(expr).__mro__:
        if clz in function_table:
            return function_table[clz](
                expr, symbol_table=symbol_table, function_table=function_table
            )
    raise NotImplementedError(f"Implementation missing for {type(expr)}")


def map_args(args: tuple[sy.Expr, ...], **kwargs) -> tuple[PyTree, ...]:
    return tuple(map(ft.partial(translate, **kwargs), args))


def binary_op(op: BinaryOp) -> Translation:
    def fun(expr, **kwargs):
        return ft.reduce(op, map_args(expr.args, **kwargs))

    return fun


def unary_op(op: UnaryOp) -> Translation:
    def fun(expr, **kwargs):
        if len(expr.args) != 1:
            raise ValueError(
                f"Trying to call unary op {type(expr)} with arguments {expr.args}"
            )
        return op(translate(expr.args[0], **kwargs))

    return fun


def type_cast(cast: Callable[[sy.Expr], PyTree]) -> Translation:
    def fun(expr, **kwargs):
        return cast(expr)

    return fun


def call_on_tuple(fn: Callable[[tuple[PyTree, ...]], PyTree]) -> Translation:
    def fun(expr, **kwargs):
        return fn(map_args(expr.args, **kwargs))

    return fun


def call(fn: Callable[[PyTree, ...], PyTree]):
    def fun(expr, **kwargs):
        return fn(*map_args(expr.args, **kwargs))

    return fun


def symbol_lookup(name: str, symbol_table: SymbolTable) -> PyTree:
    try:
        val = symbol_table[name]
    except KeyError:
        raise ValueError(f"Symbol {name} could not be found in the symbol table")
    return val


def symbol(expr: sy.Expr, symbol_table: SymbolTable, **kwargs) -> PyTree:
    return symbol_lookup(expr.name, symbol_table)
