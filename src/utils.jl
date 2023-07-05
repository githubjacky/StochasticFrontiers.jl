"""
    convert(::Type{Vector}, a::Tuple{Vararg{T}}) where T 
    convert(::Type{Vector}, a::T) where T
    convert(::Type{Matrix{T}}, a::AbstractVector) where T

*optinal utility function*

Convert Tuple or single value to Vector or convert the one dimensional Vector
to Matrix.

# Examples
```juliadoctest
julia> a = (:a, :b, :c); b = :a;

julia> convert(Vector, a)
3-element Vector{Symbol}:
 :a
 :b
 :c

julia> convert(Vector, b)
1-element Vector{Symbol}:
 :a

juila> e = [1,2,3]; convert(Matrix, e)
3×1 Matrix{Int64}:
 1
 2
 3
```

"""
convert(::Type{Vector}, a::Tuple{Vararg{T}}) where T   = collect(a)::Vector{T}
convert(::Type{Vector}, a::T) where T                  = T[a]
convert(::Type{Matrix{T}}, a::AbstractVector) where T  = reshape(a, length(a), 1)::Matrix{T}


"""
    readframe(ind::Symbol; df::DataFrame)
    readframe(ind::NTuple{N, Symbol}; df::DataFrame) where N
    readframe(ind::AbstractArray; kwargs...)

*optional utility function*

Read in data from data frame and convert it to Matrix. Besides, if it's an `AbstractArray`,
only convert to matrix

# Examples
```juliadoctest
julia> using DataFrames

julia> ivar = [1, 1, 1, 2, 2]; y=[2, 2, 4, 3, 3]; X=[3 5; 4 7; 5 9; 8 2; 3 2];

julia> md = hcat(ivar, y, X); df = DataFrame(md, [:ivar, :y, :X1, :X2]);

julia> readframe(:y, df = df)
5×1 Matrix{Int64}:
 2
 2
 4
 3
 3

julia> readframe((:X1, :X2), df = df)
5×2 Matrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  2

julia> md[:, 2]
5-element Vector{Int64}:
 2
 2
 4
 3
 3

julia> readframe(md[:, 2])
5×1 Matrix{Int64}:
 2
 2
 4
 3
 3

julia> readframe(md[:, 2:3])
5×2 Matrix{Int64}:
 2  3
 2  4
 4  5
 3  8
 3  3

```

"""
readframe(idx::Symbol; df)                           = convert(Matrix{Float64}, Vector{Float64}(Base.getproperty(df, idx)))::Matrix{Float64}
readframe(ind::Tuple{Vararg{Symbol}}; df)            = reduce(hcat, Vector{Float64}[Base.getproperty(df, i) for i in ind])::Matrix{Float64}
readframe(a::AbstractArray{T}; kwargs...) where T    = convert(Matrix{T}, a)::Matrix{T}


"""
    paramname_col1(dist_fieldname::Tuple{Symbol})

*optional utility function*

Create the first column of the output estimation table through the field names
of `<:AbstractDist`

*Warning: this method only create the base template, further completion should
be done applying model specific method `sfspec`

# Examples:
```juliadoctest
julia> dist = Trun(rand(500, 4), reshape(ones(500), 500, 1));

julia> paramname_col1(fieldnames(Trun))
4-element Vector{Symbol}:
 :frontiers
 :μ
 :log_σᵤ²
 :log_σᵥ²
```

"""
function paramname_col1(dist_fieldname)
    en   = length(dist_fieldname) + 2
    col1 = Vector{Symbol}(undef, en)

    col1[begin:en] = begin
        dist_fieldname[end] == :σᵤ² ?
            [:frontiers, dist_fieldname[begin:end-1]..., :log_σᵤ², :log_σᵥ²] :
            [:frontiers, dist_fieldname[begin:end-1]..., :log_λ², :log_σᵥ²]
    end

    return col1
end


"""
    create_names(label::Symbol, a::AbstractMatrix{<:Real})
    create_names(a::Tuple{Vararg{Symbol}})
    create_names(a::Symbol)

utility function to create the second column of the output estimation table with Esample

# Examples
```juliadoctest
julia> a = (:b, :c, :d); create_names(a)
3-element Vector{Symbol}:
 :b
 :c
 :d

julia> a = rand(2, 3); create_names(:x, a)
3-element Vector{Symbol}:
 :x1
 :x2
 :x3
```

julia> a = :x; create_names(a)
1-element Vector{Symbol}:
 :x

"""
create_names(label::Symbol, a::AbstractMatrix) = Symbol[Symbol(label, i) for i = axes(a, 2)]
create_names(a::Tuple)                         = convert(Vector, a)::Vector{Symbol}
create_names(a::Symbol)                        = convert(Vector, a)::Vector{Symbol}


"""
    paramname_col2(frontiers::Tuple{Vararg{Symbol}}, dist_props, σᵥ²)
    paramname_col2(frontiers::Matrix{<:Real}, dist_props, σᵥ²)

*optional utility function*

Create the second column of the output estimation table. The former is for frame
Data while the latter is for the matrix data

*Warning: this method only create the base template, further completion should
be done applying model specific method `sfspec`.

# Arguments
- `frontiers::Union{Tuple{Vararg{Suumbol}}, Matrix{<:Real}}`

- `dist_props::Union{Tuple{Vararg{Symbol}}, Tuple{Vararg{Matrix{T}}}} where{T<:Real}`

- `σᵥ²::Union{Union{Symbol, Tuple{Vararg{Symbol}}}, Matrix{<:Real}}`

# Esample
```juliadoctest
julia> frontiers = (:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons); 

julia> dist_props = ((:age, :school, :yr, :_cons),:_cons); σᵥ² = :cons;

juila> paramname_col2(frontiers, dist_props, σᵥ²)
50-element Vector{Vector{Symbol}}:
    [:Lland, :PIland, :Llabor, :Lbull, :Lcost, :yr, :_cons]
    [:age, :school, :yr, :_cons]
    [:cons]
    [:cons]
   ⋮
 #undef
 #undef

julia> frontiers = rand(500, 7);

julia> dist_props = (rand(500, 4), convert(Matrix{Float64}, ones(500))); 

julia> σᵥ² = convert(Matrix{Float64}, ones(500));

julia> paramname_col2(frontiers, dist_props, σᵥ²)
4-element Vector{Vector{Symbol}}:
 [:frontiers1, :frontiers2, :frontiers3, :frontiers4, :frontiers5, :frontiers6, :frontiers7]
 [:exogenous1, :exogenous2, :exogenous3, :_cons]
 [:_cons]
 [:_cons]
```

"""
function paramname_col2(frontiers::Tuple, dist_props, σᵥ²)
    en   = length(dist_props) + 2
    col2 = Vector{Vector{Symbol}}(undef, en)

    col2[1:en] = [
        create_names(frontiers), create_names.(dist_props)..., create_names(σᵥ²)
    ]

    return col2
end


function paramname_col2(frontiers::AbstractMatrix, dist_props, σᵥ²)
    en = length(dist_props) + 2
    col2 = Vector{Vector{Symbol}}(undef, en)
    col2[begin] = create_names(:frontiers, frontiers)

    for i = eachindex(dist_props)
        if numberofvar(dist_props[i]) == 1
            col2[begin+i] = [:_cons]
        else
            col2[begin+i] = create_names(:exogenous, dist_props[i])
            col2[begin+i][end] = :_cons
        end
    end

    if numberofvar(σᵥ²) == 1
        col2[en] = [:_cons]
    else
        col2[en] = create_names(:exogenous, σᵥ²)
        col2[en][end] = :_cons
    end

    return col2
end


"""
    complete_template(base, add...)

*primary utility function*

Complete the template `base` with `add`

# Examples
```juliadoctest
julia> base = paramname_col1(fieldnames(Trun))
50-element Vector{Symbol}:
    :fontiers
    :μ
    :log_σᵤ²
    :log_σᵥ²

julia> complete_template(base, :hscale, :ρ)
6-element Vector{Symbol}:
 :fontiers
 :μ
 :log_σᵤ²
 :log_σᵥ²
 :hscale
 :ρ
```

"""
function complete_template(base, add...)
    len1   = length(base)
    len2   = length(add)
    concat = similar(base, len1 + len2)

    concat[1:len1]           .= base
    concat[len1+1:len1+len2] .= add

    return concat
end



"""
    paramname(col1::Vector{Symbol}, col2::Vector{Vector{Symbol}})

*primary utility function*

Merge the ther first and second column of the output estimation table.

# Examples
```juliadoctest
julia> dist = Trun(rand(500, 4), reshape(ones(500), 500, 1));

julia> col1 = complete_template(paramname_col1(fieldnames(Trun)));

julia> frontiers = rand(500, 7); dist_props = (rand(500, 4), reshape(ones(500), 500, 1)); σᵥ²=reshape(ones(500), 500, 1);

julia> col2 = complete_template(paramname_col2(frontiers, dist_props, σᵥ²), []);

julia> paramname(col1, col2)
13×2 Matrix{Symbol}:
 :fontiers   :frontiers1
 Symbol("")  :frontiers2
 Symbol("")  :frontiers3
 Symbol("")  :frontiers4
 Symbol("")  :frontiers5
 Symbol("")  :frontiers6
 Symbol("")  :frontiers7
 :μ          :exogenous1
 Symbol("")  :exogenous2
 Symbol("")  :exogenous3
 Symbol("")  :_cons
 :log_σᵤ²    :_cons
 :log_σᵥ²    :_cons

See also: [`complete_template`](@ref)
```

"""
function paramname(col1, col2::Vector{Vector{Symbol}})
    names = Matrix{Symbol}(undef, sum(length.(col2)), 2)
    beg   = 1

    @inbounds for i = eachindex(col1)
        col2ᵢ    = col2[i]
        len      = length(col2ᵢ)
        en       = beg + len - 1
        col1ᵢ    = repeat([Symbol()], len)
        col1ᵢ[1] = col1[i]

        names[beg:en, 1] .= col1ᵢ
        names[beg:en, 2] .= col2ᵢ

        beg = en + 1
    end

    return names
end


"""
    getvar(df, type`::Type{<:AbstractEconomicType}`, dist, _σᵥ², _depvar, _frontiers)
    getvar(df, _ivar, type, dist, _σᵥ², _depvar, _frontiers)

*primary utility function*

# Arguments
- `df::Union{Nothing, DataFrame}`       : for matrix data, type of df is `Nothing`
- `type::Symbol`                        : economic interpretation
- `dist::Tuple{Vararg{Any}}`            : the first element is the `DataType` of 
                                        distribution and all the other are the fields

- `_σᵥ²::Union{Matrix{<:Real}, Union{Symbol, Tuple{Vararg{Symbol}}}}`: 
    if there is marginal effect and it's frame data for `σᵥ²`, 
    then the type fo it is `Tuple{Vararg{Symbol}}}`

- `_depvar::Union{Matrix{<:Real}, Symbol}`                   : dependent variable
- `_frontiers::Union{Matrix{<:Real}), Tuple{Vararg{Symbol}}}`: explanatory variables

- `_ivar::Union{Vector{<:Real}, Symbol}`: for panelization cross observations, 
                                        check `static_panelize` in src/types.jl
    

To ensure return type stability, apply multiple dispatch and parameters' names 
for the `output_table()` are included to return.

*Warning: this method only create the base template, further completion should
be done applying model specific method `sfspec` which should be further defined.*

# Examples
```juliadoctest
julia> y=[2, 2, 4, 3, 3]; X=[3 5; 4 7; 5 9; 8 2; 3 2];

julia> data, col1, col2 = getvar((), Prod, (Half, (ones(5),)), ones(5), y, X)
(StochasticFrontiers.Data{DataType, Half{Matrix{Float64}}, Matrix{Float64}, Matrix{Int64}, Matrix{Int64}}(Prod, Half{Matrix{Float64}}([1.0; 1.0; … ; 1.0; 1.0;;]), [1.0; 1.0; … ; 1.0; 1.0;;], [2; 2; … ; 3; 3;;], [3 5; 4 7; … ; 8 2; 3 2], 5), [:fontiers, :log_σᵤ², :log_σᵥ², #undef, #undef, #undef, #undef, #undef, #undef, #undef  …  #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef], [[:frontiers1, :frontiers2], [:_cons], [:_cons], #undef, #undef, #undef, #undef, #undef, #undef, #undef  …  #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef])

julia> ivar = [1, 1, 1, 2, 2]; md = hcat(ivar, y, X, ones(5)); frd = DataFrame(md, [:ivar, :y, :X1, :X2, :_cons]);

julia> paneldata, col1, col2 = getvar(
           (frd,), :ivar, Prod, (Half, (:_cons,)), :_cons, :y, (:X1, :X2)
       )
(StochasticFrontiers.PanelData{DataType, Half{Panel{Matrix{Float64}}}, Panel{Matrix{Float64}}, Panel{Matrix{Float64}}, Panel{Matrix{Float64}}}(Prod, Half{Panel{Matrix{Float64}}}(Any[1.0; 1.0; … ; 1.0; 1.0;;]), Any[1.0; 1.0; … ; 1.0; 1.0;;], Any[2.0; 2.0; … ; 3.0; 3.0;;], Any[3.0 5.0; 4.0 7.0; … ; 8.0 2.0; 3.0 2.0], 5), [:fontiers, :log_σᵤ², :log_σᵥ², #undef, #undef, #undef, #undef, #undef, #undef, #undef  …  #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef], [[:X1, :X2], [:_cons], [:_cons], #undef, #undef, #undef, #undef, #undef, #undef, #undef  …  #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef, #undef])
``` 

"""
function getvar(df, type::T, _dist, _σᵥ², _depvar, _frontiers, verbose) where{T<:AbstractEconomicType}
    dist_param             = unpack(_dist)
    dist                   = _dist(df)
    σᵥ², depvar, frontiers = readframe.((_σᵥ², _depvar, _frontiers), df=df)

    # check multicollinearity
    frontiers = isMultiCollinearity(:frontiers, frontiers; warn = verbose)[1]
    σᵥ²       = isMultiCollinearity(:σᵥ², σᵥ²; warn = verbose)[1]

    # creat parameters' names of the output estimation table
    col1 = paramname_col1(fieldnames(typeof(dist)))  # generate the parameters' names for making estimation table
    col2 = df isa DataFrame ?
        paramname_col2(_frontiers, dist_param, _σᵥ²) : 
        paramname_col2(frontiers, dist_param, σᵥ²)  # generate the parameters' names for making estimation table

    return Data{T}(type, σᵥ², depvar, frontiers, size(depvar, 1)), dist, col1, col2
end

function getvar(df, _ivar, type::T, _dist, _σᵥ², _depvar, _frontiers, verbose) where{T<:AbstractEconomicType}
    ivar   = df isa Nothing ? _ivar : Base.getproperty(df, _ivar)
    tnum   = [length(findall(x->x==i, ivar)) for i in unique(ivar)]
    rowidx = tnumTorowidx(tnum)
    
    dist_param             = unpack(_dist)
    dist                   = _dist(df)
    σᵥ², depvar, frontiers = readframe.((_σᵥ², _depvar, _frontiers), df=df)

    # check multicollinearity
    frontiers = isMultiCollinearity(:frontiers, frontiers; warn = verbose)[1]
    σᵥ²       = isMultiCollinearity(:σᵥ², σᵥ²; warn = verbose)[1]

    # creat parameters' names of the output estimation table
    col1 = paramname_col1(fieldnames(typeof(dist)))  # generate the parameters' names for making estimation table
    col2 = isa(df, DataFrame) ? 
        paramname_col2(_frontiers, dist_param, _σᵥ²) : 
        paramname_col2(frontiers, dist_param, σᵥ²)  # generate var parameters for making estimation table

    return PanelData{T}(rowidx, type, σᵥ², depvar, frontiers, numberofobs(depvar)), dist, col1, col2
end


"""  
    _modelinfo(modelinfo1::String, modelinfo2::String)

*primary utility function*

Print some customized model information(before optimization process)

"""
function _modelinfo(modelinfo1, modelinfo2)
    printstyled("\n*********************************\n "; color=:cyan)
    printstyled("     Model Specification      \n"; color=:cyan); 
    printstyled("*********************************\n\n"; color=:cyan)
    println("    $(modelinfo1)\n")  # name of model
    println("$(modelinfo2)")  # some customized information
end


function num(a)
    str_a = string(a)
    check = str_a[end]
    if check == '1'
        return "$(str_a) st"
    elseif check == '2'
        return "$(str_a) nd"
    elseif check == '3'
        return "$(str_a) rt"
    else
        return "$(str_a) th"
    end
end

"""
    isMultiCollinearity(name::Symbol, themat::Matrix{<:Real})
    isMultiCollinearity(_d::AbstractDist)

*optional utility function*

Two method to check the Multicollinearity. The first element of return tuple is 
non-multicollinearity matrix and the second is indices of pivot columns

"""
function isMultiCollinearity(name::Symbol, a; warn = true)
    colnum = size(a, 2)
    colnum == 1 && (return a, [-1])
    pivots = rref_with_pivots(a)[2]::Vector{Int64}

    if length(pivots) != colnum
        if warn
            printstyled("\n * Find Multicollinearity\n\n", color=:red)
            for j in filter(x->!(x in pivots), 1:colnum)
                println("    the $(num(j)) column in $(name) is dropped")
            end
        end
        return a[:, pivots], pivots
    end

    return a, pivots
end


"""
    isconstant(label::Symbol, a::Abstractarray{<:Real, N}) where N
    isconstant(a::AbstractDist; verbose::Bool = true)

Check the vector wheter is a constant. if `warn`, stop the process and 
print out the error message.

# Examples
```juliadoctest
julia> a = ones(3); isconstant(a)
1-element Vector{Float64}:
 1.0

julia> dist = Half([1 2; 3 4]); isconstant(dist)
Half{Vector{Int64}}([1])
```

"""
function isconstant(label::Symbol, a)
    size(a, 2) > 1 && error("the $label must be constant")
    constant = length(unique(a)) == 1 ? true : false
    !constant && error("the $label must be constant")

    return a
end

function isconstant(a::T) where{T<:AbstractDist}
    new_dist = T(
        isconstant.(fieldnames(T), unpack(a))...
    )

    return new_dist
end


"""
    Ψ(frontiers, fitted_dist, σᵥ²)

*primary utility function*

The complete definition is that `ψ[begin:end-1]` store the length of parameters such as `frontiers`, 
field of `<:AbstractDist` (e.g. `μ`, `σᵤ²`). Besides, `ψ[end]` store the total 
numbers of parameters alias `sum(ψ[begin:end-1]`.

*This method only create the base template, further completion should
be done applying model specific method `sfspec`.*


"""
function Ψ(frontiers, fitted_dist, σᵥ²)
    # length of 30 just for the prevention
    # notice that the type can't be assiguned(`Vector{Int}(undf, 30)` is not allowed) since we need to define elements later
    ψ = Vector{Int64}(undef, 3)
    ψ[1:3] .= numberofvar(frontiers), sum(numberofvar.(unpack(fitted_dist))), numberofvar(σᵥ²)

    return ψ
end


"""
    slice(ξ::Vector{<:Real}, ψ::Vector{Int}; mle=false)

*primary utility function*

It's a general function to slice the array given t(he length of each segament and 
ψ is the length of each part of ξ. If it's for MLE(`mle`=true) then the ψ[end] is sum(ψ[begin:end-1])

# Examples
```juliadoctest
julia> ξ = [3., 4.5, 0.8, 1.9, 2.]; ψ = [3, 1, 1];

julia> slice(ξ, ψ)
3-element Vector{Vector{Float64}}:
 [3.0, 4.5, 0.8]
 [1.9]
 [2.0]

julia> ψ = [3, 2, 5]; slice(ξ, ψ, mle=true)
2-element Vector{Vector{Float64}}:
 [3.0, 4.5, 0.8]
 [1.9, 2.0]
```

"""
function slice(ξ::Vector{T}, ψ; mle=false) where T
    p   = mle ? 
        Vector{Vector{T}}(undef, length(ψ)-1) : 
        Vector{Vector{T}}(undef, length(ψ))

    beg = 1
    for i = eachindex(p)
        en   = beg + ψ[i] - 1
        p[i] = ξ[beg:en]
        beg  = en + 1
    end

    return p
end


"""
    numberofvar(m::AbstractMatrix)    

*optional utility function*


To check the number of explanatory variables.
"""
numberofvar(m) = size(m, 2)


"""
    numberofobs(m::AbstractMatrix{<:Real})
    numberofobs(m::AbstractVector{<:Real})
    numberofobs(a::AbstractData)

*optional utility function*

To calculate the number of observations.

"""
numberofobs(m::AbstractMatrix) = size(m, 1)
numberofobs(v::AbstractVector) = length(v)
numberofobs(a::AbstractData)   = a.nofobs


"""
    unpack(A, args...)

*optional utility function*

If the `length` of `args` is 0 then all properties of the type `A` will be return, while if
it's not, partial properties are returned given the corresponding fieldnames.

# Example
```juliadoctest
julia> dist=Trun(ones(100), ones(100));

julia> unpack(dist)
2-element Vector{Vector{Float64}}:
 [1.0, 1.0, 1.0  …  1.0, 1.0]
 [1.0, 1.0, 1.0  …  1.0, 1.0]

julia> unpack(dist, :μ)
1-element Vector{Vector{Float64}}:
 [1.0, 1.0, 1.0  …  1.0, 1.0]
```

"""
function unpack(A, args...)
    if length(args) == 0
        return [Base.getproperty(A, i) for i in fieldnames(typeof(A))]
    else
        return [Base.getproperty(A, i) for i in args]
    end
end


"""
    tnumTorowidx(tnum::Vector{Int64})

Transform number of T(periods) `tnum` to `rowidx` for the further process in `sf_deman` or 
`static_panelize`. In panel data, for each individual i, we observe multiple periods.
`rowidx` is the array of `UnitRange{Int64}` to cluster the same observed individual in the data.

# Examples
```juliadoctest
julia> tnum = [2, 3, 5];

julia> rowidx = tnumTorowidx(tnum)
3-element Vector{UnitRange{Int64}}:
 1:2
 3:5
 6:10
```

"""
function tnumTorowidx(tnum)
    rowidx = Vector{UnitRange{Int}}(undef, length(tnum))
    beg = 1
    @inbounds for i in eachindex(rowidx)
        en = beg + tnum[i] - 1
        rowidx[i] = beg:en
        beg = en + 1
    end
    
    return rowidx
end


"""
    static_panelize(a::AbstractVector{<:Real}, rowidx)
    static_panelize(b::AbstractMatrix{<:Real}, rowidx)

Panelize the data into `StaticArray` for quich operation. The `rowidx` is the given information 
of the observed period for each observation i and its type is `Vecotor{UnitRange{Int64}}`.

# Examples
```juliadoctest
julia> static_panelize([1,1,1,2,2], tnumTorowidx([3, 2]))
2-element Vector{SVector}:
 [1, 1, 1]
 [2, 2]

julia> static_panelize([1 2; 1 2; 1 2; 3 4; 3 4], tnumTorowidx([3, 2]))
2-element Vector{SMatrix}:
 [1 2; 1 2; 1 2]
 [3 4; 3 4]
```

"""
function static_panelize(a::AbstractVector{T}, rowidx) where T
    return Vector{T}[
        a[i]
        for i in rowidx
    ]
end

function static_panelize(a::AbstractMatrix{T}, rowidx) where T
        return Matrix{T}[
            a[i, :]
            for i in rowidx
        ]
end


"""
    sf_demean(a::AbstractVector, rowidx)
    sf_demean(a::AbstractMatrix, rowidx)

"Demean" function for panel data.

# Examples
```juliadoctest
julia> sf_demean([1,2,3,2,4], tnumTorowidx([3, 2]))
5-element Vector{Int64}:
 -1
  0
  1
 -1
  1

julila> sf_demean([1 2; 2 3; 3 4; 5 6; 6 7], tnumTorowidx([3, 2]))
5×2 Matrix{Float64}:
 -1.0  -1.0
  0.0   0.0
  1.0   1.0
 -0.5  -0.5
  0.5   0.5
```

"""
function sf_demean(a::AbstractVector, rowidx)
    demean = similar(a)
    
    @inbounds for idx in rowidx
        instance = view(a, idx)
        demean[idx] = instance .- mean(instance)
    end
    
    return demean
end

function sf_demean(a::AbstractMatrix, rowidx)
    demean = similar(a)
    
    @inbounds for idx in rowidx
        instance = view(a, idx, :)
        demean[idx, :] = instance .- mean(instance, dims=1)
    end
    
    return demean
end


"""
    demean_panelize(a::AbstractVector{<:Real}, rowidx)
    demean_panelize(b::AbstractMatrix{<:Real}, rowidx)

panelize after demean
"""
function demean_panelize(a::AbstractVector{T}, rowidx) where T
    data = Vector{Vector{T}}(undef, length(rowidx))
    
    @inbounds for i = eachindex(data)
        instance = view(a, rowidx[i])
        data[i] = instance .- mean(instance)
    end
    
    return data
end

function demean_panelize(a::AbstractMatrix{T}, rowidx) where T
    data = Vector{Matrix{T}}(undef, length(rowidx))
    
    @inbounds for i = eachindex(data)
        instance = view(a, rowidx[i], :)
        data[i] = instance .- mean(instance, dims = 1)
    end
    
    return data
end


"""
    valid_range(range::Vector{UnitRange{Int}}, lag::Int; shift::Int = 0)

if `lag = 0`, is the case in `lagdrop` and if it's not, it's used in `lagshift`.
The main purpose is to get the target indices which is useful when there is the time 
adjustment of panel data.

# Examples
```juliadoctest
julia> rowidx = [1:3, 4:5, 6:8]; totallag = 2;

julia> valid_range(rowidx, totallag)
2-element Vector{UnitRange{Int64}}:
 3:3
 8:8

julia> valid_range(rowidx, totallag; shift = 1)
2-element Vector{UnitRange{Int64}}:
 2:2
 7:7
```

"""
function valid_range(rowidx, lag; shift = 0) 

    return UnitRange{Int64}[
        view(i, (1+lag) : length(i)) .- shift 
        for i in rowidx if length(i) > lag
    ]

end


"""
    drop_panelize(a, rowidx::Vector{UnitRange{Int64}}, lag::Int64)

The concept is similar to first `lagdrop`, then `panelize`. 

# Examples
```juliadoctest
julia> drop_panelize([1 2; 3 4; 5 6; 7 8; 9 10], tnumTorowidx([3, 2]), 1)
2-element Vector{SMatrix}:
 [3 4; 5 6]
 [9 10]
```

"""
function drop_panelize(a, rowidx, lag; shift = 0)
    _rowidx = valid_range(rowidx, lag; shift = shift)
    return static_panelize(a, _rowidx)
end


"""
    newrange(rowidx::Vector{UnitRange{Int}}, lag::Int)

update the `rowidx` when the panelization target has been `lagdrop` before

# Examples
```juliadoctest
julia> a = [1:3, 4:5, 6:8]; totallag = 2;

julia> newrange(a, totallag)
2-element Vector{UnitRange{Int64}}:
 1:1
 2:2
```

"""
function newrange(rowidx, lag) 
    @inbounds newrange = [
        rowidx[i][(begin:end-lag)] .- (i-1)*lag
        for i in eachindex(rowidx) if length(rowidx[i])>lag
    ]

    return newrange
end


"""
    lagdrop(a::AbstractVector{<:Real}, rowidx::Vector{UnitRange{Int64}}, totallag::Int64)
    lagdrop(a::AbstractMatrix{<:Real}, rowidx::Vector{UnitRange{Int64}}, totallag::Int64)

Since the serial correlation of `ϵ`, some data in the `log_σᵤ²` and `dist_param` 
should be dropped

# Examples
```juliadoctest
julia> rowidx = [1:3, 4:5];

julia> data = rand(5)
5-element Vector{Float64}:
 0.3286279462443019
 0.9627880608384104
 0.9746855536205501
 0.9559038940357367
 0.48074509604354376


julia> lagdrop(data, rowidx, 2)
1-element Vector{Float64}:
 0.9746855536205501

julia> data = [
           1  2  3  5;
           1  2  4  7;
           1  4  5  9;
           2  3  8  2;
           2  3  3  2;
       ];


julia> lagdrop(data, rowidx, 2)
1×4 Matrix{Int64}:
 1  4  5  9

```

"""
function lagdrop(a::AbstractVector, rowidx, lag)
    new = valid_range(rowidx, lag)

    return a[union(new...)]
end

function lagdrop(a::AbstractMatrix, rowidx, lag)
    new = valid_range(rowidx, lag)

    return a[union(new...), :]
end


"""
    lagshift(a::AbstractVector{<:Real}, rowidx, lag::Int64, totallag::Int64) where T
    lagshift(a::AbstractMatrix{<:Real}, rowidx, lag::Int64, totallag::Int64) where T

Get the `lag` data for autocorrelation terms which is shifted after `lagdrop`

# Examples
```juliadoctest
julia> rowidx = [1:3, 4:5];

julia> data = [
           1  2  3  5;
           1  2  4  7;
           1  4  5  9;
           2  3  8  2;
           2  3  3  2;
       ];

julia> lagdrop(data, rowidx, 2)
1×4 Matrix{Int64}:
 1  4  5  9

julia> lagshift(data, rowidx, 2; lag = 1)
1×4 Matrix{Int64}:
 1  2  4  7

julia> lagshift(data, rowidx, 2; lag = 2)
1×4 Matrix{Int64}:
 1  2  3  5
```

"""
function lagshift(a::AbstractVector, rowidx, lag; shift)
    new = valid_range(rowidx, lag; shift = shift)

    return a[union(new...)]
end

function lagshift(a::AbstractMatrix, rowidx, lag; shift)
    new = valid_range(rowidx, lag; shift = shift)

    return a[union(new...), :]
end
