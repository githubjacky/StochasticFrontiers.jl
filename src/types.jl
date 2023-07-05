# Stochastic Frontiers Model
abstract type AbstractSFmodel end
abstract type AbstractPanelModel <: AbstractSFmodel end
abstract type AbstractUndefSFmodel end


# utility function of AbstractSFmodel
"""
The type of distibution

"""
typeofdist(a::AbstractSFmodel) = typeof(a.dist)


"""
number of parameters 

"""
numberofparam(a::AbstractSFmodel) = a.ψ[end]


# economic interpredation
abstract type AbstractEconomicType end
struct Prod <: AbstractEconomicType end
struct Cost <: AbstractEconomicType end
const prod = Prod
const p    = Prod
const cost = Cost

Base.:*(::Prod, a) = a
Base.:*(::Cost, a) = -a


# Distributions assumptions
abstract type AbstractDist end

struct Half{T} <: AbstractDist
    σᵤ²::T
end

struct Trun{T, S} <: AbstractDist
    μ::T
    σᵤ²::S
end

struct Expo{T} <: AbstractDist
    λ::T
end

const half = Half
const trun = Trun
const expo = Expo


# initial construction, further completeness in `getvar`
Half(;σᵤ²)    = Half(σᵤ²)
Trun(;μ, σᵤ²) = Trun(μ, σᵤ²)
Expo(;λ)      = Expo(λ)


# for read in the data fiven the column symbol
(a::AbstractDist)(::Nothing) = a

function (a::Half)(df::DataFrame) 
    Half(
        isMultiCollinearity(:σᵤ², readframe(a.σᵤ², df = df))[1]
    )
end
function (a::Trun)(df::DataFrame) 
    Trun(
        isMultiCollinearity(:μ, readframe(a.μ, df = df))[1],
        isMultiCollinearity(:σᵤ², readframe(a.σᵤ², df = df))[1]
    )
end
function (a::Expo)(df::DataFrame) 
    Expo(
        isMultiCollinearity(:λ, readframe(a.λ, df = df))[1]
    )
end

(a::Half)(x::AbstractVector) = (exp.(a.σᵤ²*x),)

# product of the data and coefficients
function (a::Trun)(x::AbstractVector)
    n = numberofvar(a.μ)
    Wμ, Wᵤ = view(x, 1:n), view(x, n+1:length(x))
    return (a.μ * Wμ, exp.(a.σᵤ²*Wᵤ))
end

(a::Expo)(x::AbstractVector) = (exp.(a.λ * x),)




# for bootstrap
resample(A::T, ind) where{T<:AbstractDist} = T([field[ind, :] for field in unpack(A)]...)


# general data to fit model
abstract type AbstractData end

# This type is for the usage of `AbstractSFmodel`
"""
    Data(econtype::AbstractEconomicType, 
         σᵥ²::Matrix{<:Real}, 
         depvar::Matrix{<:Real}, 
         frontiers::Matrix{<:Real}, 
         nofobs::Int64
        )

"""
struct Data{T} <: AbstractData
    econtype::T
    σᵥ²::Matrix{Float64}
    depvar::Matrix{Float64}
    frontiers::Matrix{Float64}
    nofobs::Int64
end


# This type is for the usage of `AbstractPanelModel`.
"""
    PanelData(rowidx::Vector{UnitRange{Int64}}
              econtype::Type{<:AbstractEconomicType}, 
              σᵥ²::Matrix{<:Real}, 
              depvar::Matrix{<:Real}, 
              frontiers::Matrix{<:Real}, 
              nofobs::Int64
             )

`rowidx` is the information which should be provided for `sf_deman` or `static_panelize`.

"""
struct PanelData{T} <: AbstractData
    rowidx::Vector{UnitRange{Int64}}
    econtype::T
    σᵥ²::Matrix{Float64}
    depvar::Matrix{Float64}
    frontiers::Matrix{Float64}
    nofobs::Int64
end

# API to get the property of `AbstractData` more effieciet

numberofi(a::PanelData) = length(a.rowidx)
numberoft(a::PanelData) = length.(a.rowidx)

# for bootstrap
function (a::Data)(selected_row)
    return Data(
        a.econtype,
        a.σᵥ²[selected_row, :],
        a.depvar[selected_row, :],
        a.frontiers[selected_row, :],
        a.nofobs
    )
end

function (a::PanelData)(selected_row)
    return PanelData(
        a.rowidx,
        a.econtype,
        a.σᵥ²[selected_row, :],
        a.depvar[selected_row, :],
        a.frontiers[selected_row, :],
        a.nofobs
    )
end


abstract type AbstractSFresult end

# the main result, one of tthe fields of <: AbstractSFresult
struct MainSFresult{T, S, U, V}
    ξ::Vector{Float64}
    model::T
    data::S
    options::U
    jlms::Vector{Float64}
    bc::Vector{Float64}
    loglikelihood::Float64
    main_opt::V
end


struct SFresult{T<:MainSFresult, S<:AbstractSFresult}
    main_res::T
    model_res::S
end

