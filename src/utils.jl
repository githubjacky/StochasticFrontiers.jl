"""
    Base.convert(::Type{Vector}, a::Tuple{Vararg{T}}) where T 
    Base.convert(::Type{Vector}, a::T) where T
    Base.convert(::Type{Matrix}, a::Vector{T}}) where T

*optinal utility function*

Convert Tuple or single value to Vector or convert the one dimensional Vector
to Matrix.

See also: [`varvals`](@ref)
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
Base.convert(::Type{Vector}, a::Tuple{Vararg{T}}) where T = [i for i in a]
Base.convert(::Type{Vector}, a::Union{Real, Symbol}) = [a]
Base.convert(::Type{Matrix}, a::Vector) = reshape(a, length(a), 1)


"""
    readframe(ind::Symbol; df::DataFrame)
    readframe(ind::Tuple{Vararg{Symbol}}; df::DataFrame)
    readframe(ind::Array; kwargs...)

*optional utility function*

Read in data from data frame and convert it to Matrix. Besides, if it's matrix data,
only convert to matrix

# Examples
```juliadoctest
julia> using DataFrames

julia> ivar = [1, 1, 1, 2, 2]; y=[2, 2, 4, 3, 3]; X=[3 5; 4 7; 5 9; 8 2; 3 2];

julia> md = hcat(ivar, y, X); fd = DataFrame(md, [:ivar, :y, :X1, :X2]);

julia> readframe(:y, df=fd)
5×1 Matrix{Int64}:
 2
 2
 4
 3
 3

julia> readframe((:X1, :X2), df=fd)
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

julia> readframe(md[:, 2], df=())
5×1 Matrix{Int64}:
 2
 2
 4
 3
 3

julia> readframe(md[:, 2:3], df=())
5×2 Matrix{Int64}:
 2  3
 2  4
 4  5
 3  8
 3  3

See also: [`convert`](@ref)
```
"""
readframe(ind::Symbol; df::DataFrame) = convert(Matrix, Base.getindex(df, :, ind))
readframe(ind::NTuple{N, Symbol}; df::DataFrame) where N = hcat([Base.getindex(df, :, i) for i in ind]...)
readframe(ind::Array; kwargs...) = convert(Matrix, ind)


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
50-element Vector{Symbol}:
    :fontiers
    :μ
    :log_σᵤ²
    :log_σᵥ²
   ⋮
 #undef
 #undef

See also: [`Trun`](@ref), [`Half`](@ref), [`Expo`](@ref)
```
"""
function paramname_col1(dist_fieldname)
    col1 = Vector{Symbol}(undef, 50)
    en = length(dist_fieldname) + 2
    if dist_fieldname[end] == :σᵤ²
        col1[1:en] = [
            :fontiers, dist_fieldname[begin:end-1]..., :log_σᵤ², :log_σᵥ²
        ]
    else  # Exponential distribution
        col1[1:en] = [
            :fontiers, dist_fieldname[begin:end-1]..., :log_λ², :log_σᵥ²
        ]
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
create_names(label::Symbol, a::AbstractMatrix) = [Symbol(label, i) for i=axes(a, 2)]
create_names(a::Tuple) = convert(Vector, a)
create_names(a::Symbol) = convert(Vector, a)


"""
    paramname_col2(frontiers::Tuple{Vararg{Symbol}}, dist_props::Tuple{Vararg{Symbol}} σᵥ²::Union{Symbol, Tuple{Vararg{Symbol}}})
    paramname_col2(frontiers::Matrix{<:Real}, dist_props::Tuple{Vararg{Matrix{<:Real}}}, σᵥ²::Matrix{<:Real})

*optional utility function*

Create the second column of the output estimation table. The former is for frame
Data while the latter is for the matrix data

*Warning: this method only create the base template, further completion should
be done applying model specific method `sfspec`.

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

julia> frontiers = rand(500, 7); dist_props = (rand(500, 4), convert(Matrix, ones(500))); σᵥ²=convert(Matrix, ones(500));

julia> paramname_col2(frontiers, dist_props, σᵥ²)
50-element Vector{Vector{Symbol}}:
    [:frontiers1, :frontiers2, :frontiers3, :frontiers4, :frontiers5, :frontiers6, :frontiers7]
    [:exogenous1, :exogenous2, :exogenous3, :_cons]
    [:_cons]
    [:_cons]
   ⋮
 #undef
 #undef
```

See also: [`Base.convert`](@ref), [`varmatrix`](@ref)
"""
function paramname_col2(frontiers::Tuple, dist_props::Tuple, σᵥ²::Union{Symbol, Tuple{Vararg{Symbol}}})
    col2 = Vector{Vector{Symbol}}(undef, 50)
    en = length(dist_props) + 2
    col2[1:en] = [
        create_names(frontiers), create_names.(dist_props)..., create_names(σᵥ²)
    ]
    return col2
end


function paramname_col2(frontiers::AbstractMatrix, dist_props::Tuple, σᵥ²::AbstractArray)
    col2 = Vector{Vector{Symbol}}(undef, 50)
    en = length(dist_props) + 2
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
 #undef
   ⋮
 #undef
 #undef

julia> complete_template(base, :hscale, :ρ)
6-element Vector{Symbol}:
 :fontiers
 :μ
 :log_σᵤ²
 :log_σᵥ²
 :hscale
 :ρ
```

See also: [`paramname_col1`](@ref), [`paramname_col2`](@ref), [`ψ`](@ref)
"""
function complete_template(base, add...)
    beg = findfirst(i->!isassigned(base, i), 1:length(base))
    length(add) == 0 && (return base[begin:beg-1])
    en = beg + length(add) - 1
    base[beg:en] .= add
    return base[begin:en]
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
function paramname(col1, col2)
    names = Matrix{Symbol}(undef, sum(length.(col2)), 2)
    beg = 1
    for i = eachindex(col1)
        col2ᵢ = col2[i]
        len = length(col2ᵢ)
        en = beg + len - 1
        col1ᵢ = repeat([Symbol()], len)
        col1ᵢ[1] = col1[i]
        names[beg:en, 1] .= col1ᵢ
        names[beg:en, 2] .= col2ᵢ
        beg = en + 1
    end
    return names
end


"""
    getvar(df, type, dist, _σᵥ², _depvar, _frontiers)
    getvar(df, _ivar, type, dist, _σᵥ², _depvar, _frontiers)

*primary utility function*

# Arguments
- `df::Union{Tuple{}, Tuple{DataFrame}}`: for matrix data, type of df is `Tuple{}`
- `type::Symbol`: economic interpretation
- `dist::Tuple{Vararg{Any}}`: the first element is the `DataType` of distribution
  and all the other are the fields
- `_σᵥ²::Union{Matrix{<:Real}, Union{Symbol, Tuple{Vararg{Symbol}}}}`: if there
  is marginal effect and it's frame data for `σᵥ²`, then the type fo it is `Tuple{Vararg{Symbol}}}`
- `_depvar::Union{Matrix{<:Real}, Symbol}`
- `_frontiers::Union{Matrix{<:Real}), Tuple{Vararg{Symbol}}}`
- `_ivar::Union{Vector{<:Real}, Symbol}`: argument for `Panel Model` to create the
  `Panel` object
    
To ensure return type stability, apply multiple dispatch and parameters' names 
for the `output_table()` are included to return.

*Warning: this method only create the base template, further completion should
be done applying model specific method `sfspec`.*

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

See also: [`readframe`](@ref), [`paramname_col1`](@ref), [`paramname_col2`](@ref),
[`Data`](@ref), [`PanelData`](@ref)
"""
function getvar(data::Tuple,
                 type::Symbol,
                 _dist::Tuple,
                 _σᵥ²::Union{Array, Union{Symbol, Tuple}},
                 _depvar::Union{Array, Symbol},
                 _frontiers::Union{Matrix, Tuple},
                 verbose = true
                )
    
    # readin the data in Matrix form
    df = isa(data, Tuple{DataFrame}) ? data[1] : data
    dist_type, dist_param = _dist
    fitted_dist = dist_type(readframe.(dist_param, df=df)...)
    σᵥ², depvar, frontiers = readframe.((_σᵥ², _depvar, _frontiers), df=df)

    # check multicollinearity
    frontiers, _,  = isMultiCollinearity(:frontiers, frontiers, verbose)
    fitted_dist, _ = isMultiCollinearity(fitted_dist, verbose)
    σᵥ², _ = isMultiCollinearity(:σᵥ², σᵥ², verbose)

    # creat parameters' names of the output estimation table
    col1 = paramname_col1(fieldnames(dist_type))  # generate the parameters' names for making estimation table
    col2 = isa(df, DataFrame) ? paramname_col2(_frontiers, dist_param, _σᵥ²) : paramname_col2(frontiers, dist_param, σᵥ²)   # generate the parameters' names for making estimation table

    return Data(eval(type), σᵥ², depvar, frontiers, size(depvar, 1)), fitted_dist, col1, col2
end

function getvar(data::Tuple,
                _ivar::Union{Vector, Symbol},
                type::Symbol,
                _dist::Tuple,
                _σᵥ²::Union{Array, Symbol, Tuple},
                _depvar::Union{Array, Symbol},
                _frontiers::Union{Matrix, Tuple},
                verbose = true
                )
    # readin the data in Matrix form
    if isa(data, Tuple{DataFrame})
        df = data[1]
        ivar = Base.getindex(df, :, _ivar)  
    else
        df = data
        ivar = _ivar
    end
    tnum = [length(findall(x->x==i, ivar)) for i in unique(ivar)]
    dist_type, dist_param = _dist
    fitted_dist = dist_type(
        Panel.(
            readframe.(dist_param, df=df),
            tnum=tnum
        )...
    )
    σᵥ², depvar, frontiers = Panel.(
        readframe.((_σᵥ², _depvar, _frontiers), df=df),
        tnum=tnum
    )

    # check multicollinearity
    frontiers, _,  = isMultiCollinearity(:frontiers, frontiers, verbose)
    fitted_dist, _ = isMultiCollinearity(fitted_dist, verbose)
    σᵥ², _ = isMultiCollinearity(:σᵥ², σᵥ², verbose)

    # creat parameters' names of the output estimation table
    col1 = paramname_col1(fieldnames(dist_type))  # generate the parameters' names for making estimation table
    col2 = isa(df, DataFrame) ? paramname_col2(_frontiers, dist_param, _σᵥ²) : paramname_col2(frontiers, dist_param, σᵥ²)   # generate var parameters for making estimation table

    return PanelData(depvar.rowidx, eval(type), σᵥ², depvar, frontiers, numberofobs(depvar)), fitted_dist, col1, col2
end


"""  
    _modelinfo(modelinfo1::String, modelinfo2::String)

*primary utility function*

Print some customized model information(before optimization process)
"""
function _modelinfo(modelinfo1, modelinfo2)
    printstyled("\n * Model specification\n\n", color=:yellow)
    println("    $(modelinfo1)\n")  # name of model
    println("$(modelinfo2)")  # some customized information
end


"""
    isMultiCollinearity(name::Symbol, themat)
    isMultiCollinearity(name::Symbol, _themat::AbstractPanel)
    isMultiCollinearity(_d::AbstractDist)

*optional utility function*

Two method to check the Multicollinearity, for the usage of either matrix or panel data
The first element of return tuple is non-multicollinearity matrix and the second
is indices of pivot columns
"""
function isMultiCollinearity(name::Symbol, themat, verbose=true)
    colnum = size(themat, 2)
    colnum == 1 && return themat, 1
    pivots = rref_with_pivots(themat)[2]
    if length(pivots) != colnum
        if verbose
            printstyled("\n * Find Multicollinearity\n\n", color=:red)
            for j in filter(x->!(x in pivots), 1:colnum)
                println("    number $j column in $(name) is dropped")
            end
        end
        return themat[:, pivots], pivots
    end
    return themat, pivots
end

function isMultiCollinearity(name::Symbol, _themat::AbstractPanel, verbose=true)
    themat, pivots = isMultiCollinearity(name , _themat.data, verbose)
    return Panel(themat, _themat.rowidx), pivots
end

function isMultiCollinearity(_d::AbstractDist, verbose=true)
    name = fieldnames(typeof(_d))
    _themat = unpack(_d)

    res = isMultiCollinearity.(name, _themat, verbose)
    themat, pivots = zip(res...)
    d = typeof(_d)(themat...)

    return d, pivots
end


"""
    isconstant(label::Symbol, a::Vector{<:Real}; warn::Bool=false)
    isconstant(a::AbstractDist; warn::Bool=false)

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
function isconstant(label, a; warn=false)
    constant = length(unique(a)) == 1 ? true : false
    (warn && !constant) && error("the $label must be constant")

    return a
end

function isconstant(a::AbstractDist; warn=false)
    dist_type = typeof(a)
    labels = fieldnames(dist_type)
    data = unpack(a)
    new_dist = dist_type(isconstant.(labels, data; warn=warn)...)

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

See also: [`complete_template`](@ref)
"""
function Ψ(frontiers, fitted_dist, σᵥ²)
    # length of 30 just for the prevention
    # notice that the type can't be assiguned(`Vector{Int}(undf, 30)` is not allowed) since we need to define elements later
    ψ = Vector(undef, 30)
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
function slice(ξ, ψ; mle=false)
    p = mle ? Vector{Any}(undef, length(ψ)-1) : Vector{Any}(undef, length(ψ))
    beg = 1
    for i = eachindex(p)
        en = beg + ψ[i] - 1
        p[i] = ξ[beg:en]
        beg = en + 1
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
numberofobs(a::AbstractData) = getproperty(a, :nofobs)

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
