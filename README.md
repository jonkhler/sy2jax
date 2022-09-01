# sy2jax

##### Table of Contents  
- [What](#What)  
- [Usage](#Usage)  
  1. [Basics](#basics)
  2. [Matrix ops](#matrix-ops)
- [Extending](#Extending)
  1. [Augment the function table](#augment-the-function-table)
  2. [Add an expression handler](#add-an-expression-handler)
- [Credits](#Credits)
- [Feedback](#Feedback)
- [License](#License)

## What
A very simple and extendible [sympy](https://github.com/sympy/sympy) to [jax](https://github.com/google/jax) transpiler.

## Usage

### Basics

Define some sympy expression

```python
import sympy as sy

def poly(x: sy.Expr, m: sy.Expr):
    k = sy.Symbol("k", positive=True, integer=True)
    return sy.Sum(x ** k, (k, 0, m)).doit()

def fun(x: sy.Expr, m: sy.Expr):
    '''Some non-trivial function of two variables'''
    x = x + 1
    a = poly(x, m)
    b = poly(x, m + 1)
    return a / b 


x = sy.Symbol("x", positive=True, real=True)
m = sy.Symbol("m", positive=True, integer=True)

term = fun(x, m)
```

Then `term` has the symbolic output
$$\\frac{1 - \\left(x + 1\\right)^{m + 1}}{1 - \\left(x + 1\\right)^{m + 2}}$$

Let's make a jax function out of it that is compatible with `jax.vmap` and `jax.jit`.
```python
import jax
import jax.numpy as jnp

from sy2jax import to_jax

def fun_as_jax(x: jnp.ndarray, m: jnp.ndarray):
    symbol_table = {"x": x, "m": m}                 # define substitution table for symbols
    return to_jax(term, symbol_table=symbol_table)  # evaluate expression
```

That's all! It is perfectly compatible with all common jax tracers:

```python
x = jnp.arange(1, 5).astype(jnp.float32)
m = jnp.arange(1, 4)
vmapped = jax.vmap(jax.vmap(fun_as_jax, in_axes=(None, 0)), in_axes=(0, None))
jitted = jax.jit(vmapped)

print(jitted(x, m).shape)
%timeit jitted(x, m).block_until_ready()
```
which yields
```
(4, 3)
4.8 µs ± 747 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
```

### Matrix ops

The library also works with most matrix ops.
E.g. you can define
```python
x = sy.MatrixSymbol("x", 2, 2)
y = sy.MatrixSymbol("y", 2, 2)

M = sy.BlockMatrix([[x, y], [sy.ZeroMatrix(2, 2), sy.eye(2, 2)]])
```
Then `M` has the symbolic expression

$$ \\left[\\begin{matrix}x & y\\\\0 & \\left[\\begin{matrix}1 & 0\\\\0 & 1\\end{matrix}\\right]\\end{matrix}\\right]$$

Now we can simply wrap that again 

```python
def wrapped(x, y):
    return sy2jax.to_jax(M, symbol_table={"x": x, "y": y})

x_ = jnp.array([[1., 0.], [0., 1.]])
y_ = jnp.array([[14., 2.], [2., 23.]])

wrapped(x_, y_)
```
which correctly yields
```
DeviceArray([[ 1.,  0., 14.,  2.],
             [ 0.,  1.,  2., 23.],
             [ 0.,  0.,  1.,  0.],
             [ 0.,  0.,  0.,  1.]], dtype=float32)
```

## Extending

If you favorite sympy function is not available yet, you have two options:

  1. [Augment the function table](#augment-the-function-table)
  2. [Add an expression handler](#add-an-expression-handler)


### Augment the function table

When calling `to_jax` you can pass a `extra_function_table` of type [FunctionTable](https://github.com/jonkhler/sy2jax/blob/08913ced94358c6e340db3f5f116ed5ed73d00bc/sy2jax/_src/types.py#L29)
https://github.com/jonkhler/sy2jax/blob/08913ced94358c6e340db3f5f116ed5ed73d00bc/sy2jax/_src/ui.py#L14-L19

The function table is a `Mapping` (=`dict`) which maps the type of a `sympy.Expr` onto a [Translation](https://github.com/jonkhler/sy2jax/blob/08913ced94358c6e340db3f5f116ed5ed73d00bc/sy2jax/_src/types.py#L30).

In general a `Translation` is a `Callable` with signature 
```python
some_translation(expr: sympy.Expr, st: sy2jax.SymbolTable, ft: sy2jax.FunctionTable) -> sy2jax.PyTree
```

For example if we wanted to (re-)implement `sympy.Add`, we could do it as follows
```python
import functools as ft

# 1. define a translation for our expression
def add_translation(expr: sympy.Expr, st: sy2jax.SymbolTable, ft: sy2jax.FunctionTable) -> sy2jax.PyTree:
    args = tuple(ft.partial(to_jax, symbol_table=st, function_table=ft), expr.args)
    return ft.reduce(jnp.add, args)

# 2. add this translation to the extra function table when calling to_jax
extra_functions = {sympy.Add: add_translation}
to_jax(complicated_expr, symbol_table=some_symbols, extra_function_table=extra_functions)
```

As this task gets quite repetitive there are some helper functions pre-implemented. The same could be accomplished via
```python
from sy2jax.core import binary_op

extra_functions = {sympy.Add: binary_op(jnp.add)}
to_jax(complicated_expr, symbol_table=some_symbols, extra_function_table=extra_functions)
```

See https://github.com/jonkhler/sy2jax/blob/main/sy2jax/_src/core.py#L60-L124 for a range of pre-implemented helpers.

### Add an expression handler

The second option to extend the libary is adding a [handler](https://github.com/jonkhler/sy2jax/blob/08913ced94358c6e340db3f5f116ed5ed73d00bc/sy2jax/_src/core.py#L33-L57).

They are implemented via [functools.singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch) and in practice the same example above could be written as

```python
from sy2jax.core import handler

@handler.register
def _(expr: sympy.Add, st: sy2jax.SymbolTable, ft: sy2jax.FunctionTable) -> sy2jax.PyTree:
    args = tuple(ft.partial(to_jax, symbol_table=st, function_table=ft), expr.args)
    return ft.reduce(jnp.add, args)
```

You can see the pre-implemented functions here https://github.com/jonkhler/sy2jax/blob/08913ced94358c6e340db3f5f116ed5ed73d00bc/sy2jax/_src/default.py#L46-L92.

__Feel free to open a PR if you want to add an implementation!__

## Credits

The whole idea for this library was started, when playing around with [Patrick Kidger's](https://github.com/patrick-kidger) awesome [sympy2jax](https://github.com/google/sympy2jax) implementation.

The major reasons for writing my own library were 
1. curiosity
2. simplifying dependencies (no [Equinox](https://github.com/patrick-kidger/equinox) dependency)
3. extendiblity without breaking the library for custom sympy expressions

## Feedback

If you want to add your own expression implementation just open a PR. If you find a bug just open an issue. 

For anything else just write me on my [Twitter](https://twitter.com/jonkhler).


## License

MIT
