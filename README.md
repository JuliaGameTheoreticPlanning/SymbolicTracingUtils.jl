# SymbolicTracingUtils

**SymbolicTracingUtils** is a lightweight abstraction layer designed to simplify switching between symbolic tracing packages like [`Symbolics.jl`](https://github.com/JuliaSymbolics/Symbolics.jl) and [`FastDifferentiation.jl`](https://github.com/YingboMa/FastDifferentiation.jl). 

Unlike other abstraction layers such as `DifferentiationInterface.jl`, this package is not limited to automatic differentiation use cases but also supports scenarios where symbolic tracing is used to generate efficient implementations of user-defined functions.

## Features

- Seamless integration with both `Symbolics.jl` and `FastDifferentiation.jl`.
- Consistent APIs for creating symbolic variables, building callable functions, and computing gradients or Jacobians.
- Support for both dense and sparse Jacobian computation.
- Lightweight and minimalistic design.

## Installation

To install `SymbolicTracingUtils`, use the Julia package manager:

```julia
using Pkg
Pkg.add("SymbolicTracingUtils")
```

## Usage

### Importing the Package

```julia
using SymbolicTracingUtils
```

### Creating Symbolic Variables

```julia
backend = SymbolicsBackend()
x = make_variables(backend, :x, 3)  # Creates a 3-element vector of symbolic variables `x[1]`, `x[2]`, `x[3]`
```

### Building Callable Functions

```julia
f_symbolic = x[1]^2 + x[2]*x[3]
f_callable = build_function(f_symbolic, x; in_place = false)
result = f_callable([1.0, 2.0, 3.0])  # Evaluate the function
```

### Computing Gradients

```julia
grad = gradient(f_symbolic, x)
```

### Computing Jacobians

```julia
f_vector = [x[1]^2, x[2]*x[3]]
jac = jacobian(f_vector, x)  # Dense Jacobian
sparse_jac = sparse_jacobian(f_vector, x)  # Sparse Jacobian
```

## Exports

The following symbols are exported:

- `SymbolicsBackend`
- `FastDifferentiationBackend`
- `make_variables`
- `build_function`
- `gradient`
- `jacobian`
- `sparse_jacobian`

## License

This project is licensed under the MIT License.

---

Contributions, bug reports, and feature requests are welcome!
