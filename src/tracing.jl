"""
    make_variables(backend, name, dimension)

Creates a vector of `dimension` where each element is a scalar symbolic variable from `backend` with the given `name`.
"""
function make_variables end

function make_variables(::SymbolicsBackend, name::Symbol, dimension::Int)
    vars = Symbolics.@variables($name[1:dimension]) |> only |> Symbolics.scalarize

    if isempty(vars)
        vars = Symbolics.Num[]
    end

    vars
end

function make_variables(::FastDifferentiationBackend, name::Symbol, dimension::Int)
    FD.make_variables(name, dimension)
end

"""
    build_function(f_symbolic, args_symbolic...; in_place, options)

Builds a callable function from a symbolic expression `f_symbolic` with the given `args_symbolic` as arguments.

Depending on the `in_place` flag, the function will be built as in-place `f!(result, args...)` or out-of-place variant `restult = f(args...)`.

`backend_options` will be forwarded to the backend specific function and differ between backends.
"""
function build_function end

# scalar fallback
function build_function(f_symbolic::T, args_symbolic...; backend_options = (;)) where {T<:FD.Node}
    f_callable! =
        build_function([f_symbolic], args_symbolic...; in_place = true, backend_options...)

    let output = [0.0]
        function (x)
            f_callable!(output, x)
            only(output)
        end
    end
end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:Symbolics.Num}
    f_callable, f_callable! = Symbolics.build_function(
        f_symbolic,
        args_symbolic...;
        expression = Val{false},
        # slightly saner defaults...
        (; parallel = Symbolics.ShardedForm(), backend_options...)...,
    )

    in_place ? f_callable! : f_callable
end

function build_function(
    f_symbolic::AbstractArray{T},
    args_symbolic...;
    in_place,
    backend_options = (;),
) where {T<:FD.Node}
    FD.make_function(f_symbolic, args_symbolic...; in_place, backend_options...)
end
