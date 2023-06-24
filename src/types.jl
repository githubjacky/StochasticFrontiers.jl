# Stochastic Frontiers Model
abstract type AbstractSFmodel end
abstract type AbstractPanelModel <: AbstractSFmodel end

# some API to get some base properties of `<:SFmodel`
distof(a::AbstractSFmodel)          = getproperty(a, :fitted_dist)
typeofdist(a::AbstractSFmodel)      = typeof(getproperty(a, :fitted_dist))
get_paramlength(a::AbstractSFmodel) = getproperty(a, :ψ)
get_paramname(a::AbstractSFmodel)   = getproperty(a, :paramnames)
numberofparam(a::AbstractSFmodel)   = get_paramlength(a)[end]


# economic interpredation
abstract type AbstractEconomicType end
struct Prod <: AbstractEconomicType end
struct Cost <: AbstractEconomicType end
const prod = Prod
const p    = Prod
const cost = Cost

Base.:*(::Type{Prod}, a) = a
Base.:*(::Type{Cost}, a) = -a


"""
    tnumTorowidx(tnum::Vector{Int})

Transform number of T(periods) `tnum` to `rowidx` for the construction of type `Panel` or 
`Panelized`. In terms of panel data, for each individual i, we observe multiple periods.
`rowidx` is the array of `UnitRange` to record the index of individual i.

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

abstract type AbstractPanel{T, N} <: AbstractArray{T, N} end
struct PanelMatrix{T} <: AbstractPanel{T, 2}
    data::Matrix{T}
    rowidx::Vector{UnitRange{Int}}
end

struct PanelVector{T} <: AbstractPanel{T, 1}
    data::Vector{T}
    rowidx::Vector{UnitRange{Int}}
end

# Serve the AbstractArray interface for `Panel`
Base.size(A::AbstractPanel) = size(A.data)
function Base.getindex(A::AbstractPanel, I...)
    return Base.getindex(A.data, I...)
end


"""
    Panelized(data::Vector{AbstractVecOrMat{T}} where T, rowidx::Vector{UnitRange{Int64}})

Create vector form of panel data.

See also: [`Panel`](@ref)
"""
struct Panelized{T} <: AbstractArray{eltype(T), 1}
    data::T
    rowidx::Vector{UnitRange{Int}}
end

# Serve the AbstractArray interface for `Panelized`
Base.size(A::Panelized) = size(A.data)
function Base.getindex(A::Panelized, I...)
    return Base.getindex(A.data, I...)
end


# some utility function for `Panel` and `Panelized`
numberofi(a::AbstractPanel) = length(a.rowidx)
numberofi(a::Panelized)     = length(a.rowidx)

numberoft(a::AbstractPanel) = [length(i) for i in a.rowidx]
numberoft(a::Panelized)     = [length(i) for i in a.rowidx]



"""
    Panel(a::AbstractVector, rowidx::Vector{UnitRange{Int}})
    Panel(a::AbstractMatrix, rowidx::Vector{UnitRange{Int}})
    Panel(a::AbstractVector; tum::Vector{Int})
    Panel(a::AbstractMatrix; tum::Vector{Int})
    Panel(a::Panelized)
    
Create a `Matrix` or  `Vector` object for the usage in panel model.
There are two acceptable from of rowinfo:
1. given the `rowidx::Vector{UnitRange{Int}}`
2. given the each time period of i(keyword argument: `tnum`)

# Examples
```juliadoctest
julia> ivar = [1, 1, 1, 2, 2];

julia> y = [2, 2, 4, 3, 3]; 

julia> X = [3 5; 4 7; 5 9; 8 2; 3 2]
5×2 Matrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  2

julia> tnum = [length(findall(x->x==i, ivar)) for i in unique(ivar)]
2-element Vector{Int64}:
 3
 2

julia> data = Panel(X, tnum=tnum)
5×2 Main.SFrontiers.PanelMatrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  2

julia> mean_data = mean.(Panelized(data), dims=1)
2-element Vector{Matrix{Float64}}:
 [4.0 7.0]
 [5.5 2.0]

julia> noft = numberoft(data); 

julia> _data = [repeat(i,t) for (i,t) in zip(mean_data, noft)]
2-element Vector{Matrix{Float64}}:
 [4.0 7.0; 4.0 7.0; 4.0 7.0]
 [5.5 2.0; 5.5 2.0]

julia> Panel(reduce(vcat, _data), data.rowidx)
5×2 Main.SFrontiers.PanelMatrix{Float64}:
 4.0  7.0
 4.0  7.0
 4.0  7.0
 5.5  2.0
 5.5  2.0
```

"""
Panel(a::AbstractVector, rowidx) = PanelVector{eltype(a)}(a, rowidx)
Panel(a::AbstractMatrix, rowidx) = PanelMatrix{eltype(a)}(a, rowidx)
Panel(a::AbstractVector; tnum)   = PanelVector{eltype(a)}(a, tnumTorowidx(tnum))
Panel(a::AbstractMatrix; tnum)   = PanelMatrix{eltype(a)}(a, tnumTorowidx(tnum))
Panel(a::Panelized)              = Panel(reduce(vcat, a), a.rowidx)


# arithmetic rules
Base.:*(a::Number, b::AbstractPanel)        = Panel(a*b.data, b.rowidx)
Base.:*(a::PanelMatrix, b::AbstractVector)  = PanelVector(a.data*b, a.rowidx)
Base.:+(a::AbstractPanel, b::AbstractPanel) = Panel(a.data+b.data, a.rowidx)
Base.:-(a::AbstractPanel, b::AbstractPanel) = Panel(a.data-b.data, a.rowidx)
Broadcast.broadcast(f, a::AbstractPanel)    = Panel(broadcast(f, a.data), a.rowidx)
Broadcast.broadcast(f, a::AbstractPanel, b) = Panel(broadcast(f, a.data, b), a.rowidx)
Broadcast.broadcast(f, a, b::AbstractPanel) = Panel(broadcast(f, a, b.data), b.rowidx)


# various constructor of the `Panelized` data type
function Panelized(data::PanelMatrix)
    @inbounds res = Panelized(
        [data[i, :] for i in data.rowidx], 
        data.rowidx
    )
    return res
end

function Panelized(data::PanelVector)
    @inbounds res = Panelized(
        [data[i] for i in data.rowidx], 
        data.rowidx
    )
    return res
end


"""
    sf_demean(data::Panelized)
    sf_demean(data::AbstractPanel)

"Demean" function for panel data and no matter what input type is, the output will always
be `Panel` not `Panelized`.

# Examples
```juliadoctest
julia> ivar = [1, 1, 1, 2, 2]; 

julia> y = [2, 2, 4, 3, 3]; 

julia> X = [3 5; 4 7; 5 9; 8 2; 3 2]
5×2 Matrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  2

julia> tnum = [length(findall(x->x==i, ivar)) for i in unique(ivar)]; 

julia> data = Panel(X, tnum=tnum)
5×2 StochasticFrontiers.PanelMatrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  2

julia> sf_demean(data)
5×2 StochasticFrontiers.PanelMatrix{Float64}:
 -1.0  -2.0
  0.0   0.0
  1.0   2.0
  2.5   0.0
 -2.5   0.0
```

"""
function sf_demean(data::Panelized)
    means = mean.(data, dims=1)
    transformed = [data[i] .- means[i] for i in eachindex(data, means)]
    panel_data = Panel(reduce(vcat, transformed), data.rowidx)

    return panel_data
end
sf_demean(data::AbstractPanel) = sf_demean(Panelized(data))


"""
    fixrange(range::Vector{UnitRange{Int}}, totallag::Int)
    fixrange(range::Vector{UnitRange{Int}}, lag::Int, totallag::Int)

The former method is used by `lagdrop` and the letter is used by `lagshift`.
The main purpose is to get the target indices which is useful when there is the time 
adjustment of panel data.

# Examples
```juliadoctest
julia> a = [1:3, 4:5, 6:8]; totallag = 2;

julia> fixrange(a, totallag)
2-element Vector{UnitRange{Int64}}:
 3:3
 8:8

julia> fixrange(a, 1, totallag)
2-element Vector{UnitRange{Int64}}:
 2:2
 7:7
```

"""
function fixrange(range, totallag) 
    @inbounds newrange = [
        i[(begin+totallag):end] 
        for i in range if length(i)>totallag
    ]
    return newrange
end

function fixrange(range, lag, totallag)
    @inbounds newrange = [
        (i[(begin+totallag):end]) .- lag 
        for i in range if length(i)>totallag
    ]
    return newrange
end


"""
    newrange(range::Vector{UnitRange{Int}}, totallag::Int)

Used by `lagdrop` and `lagshift` to reset the indices of the field `rowidx`

# Examples
```juliadoctest
julia> a = [1:3, 4:5, 6:8]; totallag = 2;

julia> newrange(a, totallag)
2-element Vector{UnitRange{Int64}}:
 1:1
 2:2
```

"""
function newrange(range, totallag) 
    @inbounds newrange = [
        range[i][(begin:end-totallag)] .- (i-1)*totallag
        for i in eachindex(range) if length(range[i])>totallag
    ]

    return newrange
end


"""
    lagdrop(_data::PanelMatrix{T}, totallag::Int) where T
    lagdrop(_data::PanelVector{T}, totallag::Int) where T
    lagdrop(_data::Panelized, totallag::Int)

Since the serial correlation of `ϵ`, some data in the `log_σᵤ²` and `dist_param` 
should be dropped

# Examples
```juliadoctest
julia> _data = Panel(
           [
               1  2  3  5;
               1  2  4  7;
               1  4  5  9;
               2  3  8  2;
               2  3  3  2;
           ],
           tnum=[3, 2]
        );

julia> lagdrop(_data, 2)
1×4 StochasticFrontiers.PanelMatrix{Int64}:
 1  4  5  9

julia> _panelized_data = Panelized(_data)
2-element StochasticFrontiers.Panelized{Vector{Matrix{Int64}}}:
 [1 2 3 5; 1 2 4 7; 1 4 5 9]
 [2 3 8 2; 2 3 3 2]

julia> lagdrop(_panelized_data, 2)
1-element Panelized{Vector{Matrix{Int64}}}:
 [1 4 5 9]
```
"""
function lagdrop(_data::PanelMatrix, totallag)
    data = PanelMatrix(
        _data[union(fixrange(_data.rowidx, totallag)...), :], 
        newrange(_data.rowidx, totallag)
    )
    return data
end

function lagdrop(_data::PanelVector, totallag)
    data = PanelVector(
        _data[union(fixrange(_data.rowidx, totallag)...)], 
        newrange(_data.rowidx, totallag)
    )
    return data
end

function lagdrop(_data::Panelized, totallag)
    data = Panelized(lagdrop(Panel(_data), totallag))
    return data
end


"""
    lagshift(_data::PanelMatrix{T}, lag::Int, totallag::Int) where T
    lagshift(_data::PanelVector{T}, lag::Int, totallag::Int) where T
    lagshift(_data::Panelized{T}, lag::Int, totallag::Int) where T

Get the `lag` data for autocorrelation terms

# Examples
```juliadoctest
julia> _data = Panel(
           [
               1  2  3  5;
               1  2  4  7;
               1  4  5  9;
               2  3  8  2;
               2  3  3  2;
           ],
           tnum=[3, 2]
        );

julia> lagshift(_data, 1, totallag=2)
1×4 StochasticFrontiers.PanelMatrix{Int64}:
 1  2  4  7

julia> _panelized_data = Panelized(_data)
2-element StochasticFrontiers.Panelized{Vector{Matrix{Int64}}}:
 [1 2 3 5; 1 2 4 7; 1 4 5 9]
 [2 3 8 2; 2 3 3 2]

julia> lagshift(_panelized_data, 2, totallag=2)
1-element StochasticFrontiers.Panelized{Vector{Matrix{Int64}}}:
 [1 2 3 5]

```
"""
function lagshift(_data::PanelMatrix, lag; totallag)
    data = PanelMatrix(
        _data[union(fixrange(_data.rowidx, lag, totallag)...), :], 
        newrange(_data.rowidx, totallag)
    )
    return data
end

function lagshift(_data::PanelVector, lag; totallag)
    data = PanelVector(
        _data[union(fixrange(_data.rowidx, lag, totallag)...)], 
        newrange(_data.rowidx, totallag)
    )
    return data
end

function lagshift(_data::Panelized, lag; totallag)
    data = Panelized(lagshift(Panel(_data), lag, totallag=totallag))
    return data
end


# Distributions assumptions
abstract type AbstractDist end
struct Half{T<:AbstractArray} <: AbstractDist
    σᵤ²::T
end

struct Trun{T<:AbstractArray, S<:AbstractArray}  <: AbstractDist
    μ::T
    σᵤ²::S
end

struct Expo{T<:AbstractArray} <: AbstractDist
    λ::T
end

# initial construction, further completeness in `getvar`
Half(;σᵤ²)    = (Half, (σᵤ²,))
Trun(;μ, σᵤ²) = (Trun, (μ, σᵤ²))
Expo(;λ)      = (Expo, (λ,))

const half = Half
const trun = Trun
const expo = Expo


# product of the data and coefficients
function (s::Half)(x::Vector)
    return (broadcast(exp, s.σᵤ²*x),)
end

function (s::Trun)(x::Vector)
    μ, σᵤ² = unpack(s)
    n = numberofvar(μ)
    @inbounds Wμ, Wᵤ = x[1:n], x[n+1:end] 
    return (μ * Wμ, broadcast(exp, σᵤ²*Wᵤ))
end

function (s::Expo)(x::Vector)
    return (broadcast(exp, s.λ * x),)
end


# general data to fit model
abstract type AbstractData end

# This type is for the usage of `AbstractSFmodel`
struct Data{T<:DataType,
            U<:AbstractMatrix,
            V<:AbstractMatrix,
            W<:AbstractMatrix
           } <: AbstractData
    econtype::T
    σᵥ²::U
    depvar::V
    frontiers::W
    nofobs::Integer
end


# This type is for the usage of `AbstractPanelModel`.
struct PanelData{T<:DataType,
                 U<:PanelMatrix,
                 V<:PanelMatrix,
                 W<:PanelMatrix
                } <: AbstractData
    rowidx::Vector{UnitRange{Int}}
    econtype::T
    σᵥ²::U
    depvar::V
    frontiers::W
    nofobs::Integer
end

# API to get the property of `AbstractData` more effieciet
get_rowidx(a::PanelData)      = getproperty(a, :rowidx)
variance(a::AbstractData)     = getproperty(a, :σᵥ²)
dependentvar(a::AbstractData) = getproperty(a, :depvar)
frontier(a::AbstractData)     = getproperty(a, :frontiers)

numberofi(a::PanelData) = length(getproperty(a, :rowidx))
numberoft(a::PanelData) = length.(getproperty(a, :rowidx))

# for bootstrap
function (a::Data)(selected_row)
    econtype, σᵥ², depvar, frontiers, nofobs = unpack(a)
    bootstrap_data = Data(
        econtype,
        σᵥ²[selected_row, :],
        depvar[selected_row, :],
        frontiers[selected_row, :],
        nofobs
    )
    return bootstrap_data
end

function (a::PanelData)(selected_row)
    rowidx, econtype, σᵥ², depvar, frontiers, nofobs = unpack(a)
    bootstrap_data = PanelData(
        rowidx,
        econtype,
        Panel(σᵥ²[selected_row, :], rowidx),
        Panel(depvar[selected_row, :], rowidx),
        Panel(frontiers[selected_row, :], rowidx),
        nofobs
    )
    return bootstrap_data
end
