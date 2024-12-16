"""
Minimal abstraction on top of symbolic tracing packages such as `Symbolics.jl` and `FastDifferentiation.jl` to make switching between the two easier.

In contrast to other abstraction layers, such as DifferentiationInterface.jl, this package does not only target automatic differentiation use-cases but
also settings where tracing is performed merely to generate an efficient implementation of a user-defined function.
"""
module SymbolicTracingUtils

using Symbolics: Symbolics
using FastDifferentiation: FastDifferentiation as FD
using SparseArrays: SparseArrays

export SymbolicsBackend,
    FastDifferentiationBackend,
    make_variables,
    build_function,
    gradient,
    jacobian,
    sparse_jacobian,
    SparseFunction,
    get_constant_entries,
    get_result_buffer

struct SymbolicsBackend end
struct FastDifferentiationBackend end

include("tracing.jl")
include("derivatives.jl")
include("sparsity.jl")

end # module SymbolicTracingUtils
