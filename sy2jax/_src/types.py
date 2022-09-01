from typing import TypeVar, Callable, Mapping, Any
from numbers import Number

import jax.numpy as jnp
import sympy as sy


__all__ = [
    "PyTree",
    "Numeric",
    "UnaryOp",
    "BinaryOp",
    "SympyExprType",
    "SymbolTable",
    "FunctionTable",
    "Translation",
]


PyTree = Any
Numeric = jnp.ndarray | Number

UnaryOp = Callable[[PyTree], PyTree]
BinaryOp = Callable[[PyTree, PyTree], PyTree]

SympyExprType = TypeVar("SympyExprType", Callable, sy.Expr)

SymbolTable = Mapping[str, Numeric]
FunctionTable = Mapping[SympyExprType, "Translation"]
Translation = Callable[[sy.Expr, SymbolTable, FunctionTable], PyTree]
