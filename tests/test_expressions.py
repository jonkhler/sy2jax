import sympy as sy
import jax
import jax.numpy as jnp

import sy2jax


def test_matrix_ops():
    x = sy.MatrixSymbol("x", 2, 2)
    y = sy.MatrixSymbol("y", 2, 2)

    term = (x.T @ y).det() * y

    def wrapped(x, y):
        return sy2jax.to_jax(term, symbol_table={"x": x, "y": y})

    x_ = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    y_ = jnp.array([[14.0, 2.0], [2.0, 23.0]])

    assert jnp.allclose(wrapped(x_, y_), jnp.linalg.det(x_.T @ y_) * y_)


def test_vmap():
    x = sy.Symbol("x", positive=True, real=True)
    term = x ** 2 + 1 / (x - 1)

    @jax.vmap
    def wrapped(x):
        return sy2jax.to_jax(term, symbol_table={"x": x})

    xs = jnp.array([1.0, 2.0, 3.0])

    for x_, numeric in zip(xs, wrapped(xs)):
        symbolic = float(term.evalf(subs={x: x_}))
        assert jnp.allclose(symbolic, numeric)


def test_fraction_function_expression():
    def poly(x: sy.Expr, m: sy.Expr):
        k = sy.Symbol("k", positive=True, integer=True)
        return sy.Sum(x ** k, (k, 0, m)).doit()

    def fun(x: sy.Expr, m: sy.Expr):
        x = x + 1
        a = poly(x, m)
        b = poly(x, m + 1)
        return a / b

    x = sy.Symbol("x", positive=True, real=True)
    m = sy.Symbol("m", positive=True, integer=True)
    term = sy.integrate(fun(x, 1), x)

    xs = jnp.array([1.0, 2.0, 3.0])
    ms = jnp.array([2, 2, 3])

    @jax.jit
    def wrapped(x, m):
        return sy2jax.to_jax(term, symbol_table={"x": x, "m": m})

    for x_, m_, numeric in zip(xs, ms, wrapped(xs, ms)):
        symbolic = float(term.evalf(subs={x: x_, m: m_}))
        assert jnp.allclose(symbolic, numeric)
