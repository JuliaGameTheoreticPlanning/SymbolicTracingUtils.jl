"""
Minimal abstraction on top of symbolic tracing packages such as `Symbolics.jl` and `FastDifferentiation.jl` to make switching between the two easier.

In contrast to other abstraction layers, such as DifferentiationInterface.jl, this package does not only target automatic differentiation use-cases but
also settings where tracing is performed merely to generate an efficient implementation of a user-defined function.
"""
module SymbolicTracingUtils

using Symbolics: Symbolics
using FastDifferentiation: FastDifferentiation as FD
using SparseArrays: SparseArrays

export build_function,
    FastDifferentiationBackend,
    get_constant_entries,
    get_result_buffer,
    gradient,
    infer_backend,
    jacobian,
    make_variables,
    sparse_jacobian,
    SparseFunction,
    SymbolicNumber,
    SymbolicsBackend

struct SymbolicsBackend end
struct FastDifferentiationBackend end
const SymbolicNumber = Union{Symbolics.Num,FD.Node}
infer_backend(v::Union{Symbolics.Num,AbstractArray{<:Symbolics.Num}}) = SymbolicsBackend()
infer_backend(v::Union{FD.Node,AbstractArray{<:FD.Node}}) = FastDifferentiationBackend()

include("tracing.jl")
include("derivatives.jl")
include("sparsity.jl")

end # module SymbolicTracingUtils
