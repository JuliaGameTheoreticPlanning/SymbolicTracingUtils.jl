module SymbolicsExt

using Symbolics: Symbolics
struct SymbolicsBackend end

function make_variables(::SymbolicsBackend, name::Symbol, dimension::Int)
  vars = Symbolics.@variables($name[1:dimension]) |> only |> Symbolics.scalarize

  if isempty(vars)
    vars = Symbolics.Num[]
  end

  vars
end

function build_function(
  f_symbolic::AbstractArray{T},
  args_symbolic...;
  in_place,
  backend_options=(;),
) where {T<:Symbolics.Num}
  f_callable, f_callable! = Symbolics.build_function(
    f_symbolic,
    args_symbolic...;
    expression=Val{false},
    # slightly saner defaults...
    (; parallel=Symbolics.ShardedForm(), backend_options...)...,
  )
  in_place ? f_callable! : f_callable
end

function gradient(f_symbolic::T, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
  Symbolics.gradient(f_symbolic, x_symbolic)
end

function jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
  Symbolics.jacobian(f_symbolic, x_symbolic)
end

function sparse_jacobian(f_symbolic::Vector{T}, x_symbolic::Vector{T}) where {T<:Symbolics.Num}
  Symbolics.sparsejacobian(f_symbolic, x_symbolic)
end

end
