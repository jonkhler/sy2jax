import jax
import jax.numpy as jnp
import sympy as sy
import sympy.matrices.expressions.matexpr as matexpr

from .types import PyTree, SymbolTable, FunctionTable, Numeric
from .core import type_cast, unary_op, binary_op, call_on_tuple, call, symbol, handler


_SIMPLE_UNARY_FUNCTIONS = [
    "cbrt",
    "conjugate",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]


def _identity(x):
    return x


def _matmul(x: Numeric, y: Numeric) -> jnp.ndarray:
    """Fixes inconsistency that scalar * matrix
       is implemented as matmul within sympy."""
    if len(x.shape) == 0 or len(y.shape) == 0:
        return x * y
    return jnp.matmul(x, y)


_SIMPLE_BINARY_FUNCTIONS = [
    "gcd",
    "lcm",
]


DEFAULT_FUNCTION_TABLE: FunctionTable = {
    sy.Float: type_cast(float),
    sy.Integer: type_cast(int),
    sy.Rational: type_cast(float),
    # unary ops
    sy.Abs: unary_op(jnp.abs),
    sy.Determinant: unary_op(jnp.linalg.det),
    sy.Trace: unary_op(jnp.trace),
    sy.Transpose: unary_op(jnp.transpose),
    # automatically add all simple unary functions
    **{getattr(sy, fn): unary_op(getattr(jnp, fn)) for fn in _SIMPLE_UNARY_FUNCTIONS},
    # automatically add all inverse unary functions where defined
    **{
        getattr(sy, "a" + fn): unary_op(getattr(jnp, "arc" + fn))
        for fn in _SIMPLE_UNARY_FUNCTIONS
        if hasattr(jnp, "arc" + fn) and hasattr(sy, "a" + fn)
    },
    # binary ops
    sy.Add: binary_op(jnp.add),
    sy.MatMul: binary_op(_matmul),
    sy.MatPow: binary_op(jnp.linalg.matrix_power),
    sy.Mul: binary_op(jnp.multiply),
    sy.Pow: binary_op(jnp.power),
    # automatically add all simple binary functions
    **{getattr(sy, fn): binary_op(getattr(jnp, fn)) for fn in _SIMPLE_BINARY_FUNCTIONS},
    sy.Symbol: symbol,
    sy.MatrixSymbol: symbol,
    sy.ZeroMatrix: call_on_tuple(jnp.zeros),
    sy.OneMatrix: call_on_tuple(jnp.ones),
    sy.BlockDiagMatrix: call(jax.scipy.linalg.block_diag),
    sy.BlockMatrix: call(_identity),
}


@handler.register
def _(expr: sy.ImmutableDenseMatrix, **kwargs) -> PyTree:
    n, m, elem = expr.args
    elem = list(map(jnp.atleast_2d, map_args(elem, **kwargs)))
    return jnp.block([[a for a in elem[i::m]] for i in range(m)]).T


@handler.register
def _(expr: matexpr.MatrixElement, *, symbol_table: SymbolTable, **kwargs) -> PyTree:
    name = str(expr.args[0])
    coord = tuple(map(int, expr.args[1:]))
    mat = symbol_lookup(name, symbol_table)
    return mat[coord]
