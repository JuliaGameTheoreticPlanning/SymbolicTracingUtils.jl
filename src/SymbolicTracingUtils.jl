"""
Minimal abstraction on top of symbolic tracing packages such as `Symbolics.jl` and `FastDifferentiation.jl` to make switching between the two easier.

In contrast to other abstraction layers, such as DifferentiationInterface.jl, this package does not only target automatic differentiation use-cases but
also settings where tracing is performed merely to generate an efficient implementation of a user-defined function.
"""
module SymbolicTracingUtils

export SymbolicsBackend, FastDifferentiationBackend, make_variables, build_function

struct SymbolicsBackend end
struct FastDifferentiationBackend end

"""
    make_variables(backend, name, dimension)

Creates a vector of `dimension` where each element is a scalar symbolic variable from `backend` with the given `name`.
"""
function make_variables end

"""
    build_function(backend, f_symbolic, args_symbolic...; in_place, options)

Builds a callable function from a symbolic expression `f_symbolic` with the given `args_symbolic` as arguments.

Depending on the `in_place` flag, the function will be built as in-place `f!(result, args...)` or out-of-place variant `restult = f(args...)`.

`backend_options` will be forwarded to the backend specific function and differ between backends.
"""
function build_function end

"""
    gradient(f_symbolic, x_symbolic)

Computes the symbolic gradient of `f_symbolic` with respect to `x_symbolic`.
"""
function gradient end

"""
    jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic`.
"""
function jacobian end

"""
    sparse_jacobian(f_symbolic, x_symbolic)

Computes the symbolic Jacobian of `f_symbolic` with respect to `x_symbolic` in a sparse format.
"""
function sparse_jacobian end

end # module SymbolicTracingUtils
