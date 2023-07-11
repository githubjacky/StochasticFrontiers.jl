#########################################################################################
# TODO: create the type for models
#########################################################################################
# 1. dist, ψ, paramnames are three necessary properties

# serial correlation
abstract type AbstractSerialCorr end
struct AR <: AbstractSerialCorr
    p::Integer
end

struct MA <: AbstractSerialCorr
    q::Integer
end

struct ARMA <: AbstractSerialCorr
    p::Integer
    q::Integer
end


"""
    SNCre(fitted_dist, ψ, paramnames, serialcorr, R, σₑ², xmean)

# Arguments
- `fitted_dist::AbstractDist`     : distibution assumption of the inefficiency

- `ψ::Vector{Any}`                : record the length of each parameter, `ψ[end]` is the 
                                  arrgregate length of all parameters

- `paramnames::Matrix{Symbol}`    : parameters' names used by the output estimation table
- `SCE::AbstractSerialCorr`       : serial correlation
- `R::Int`                        : number of simulation of error of controled random effect `e`
- `σₑ²::Real`                     : vaiance of the `e`
- `xmean::AbstractMatrix{<:Real}` : explanatory variables of the controled random effect

"""
struct SNCre{T, S} <: AbstractPanelModel
    dist::T
    ψ::Vector{Int64}
    paramnames::Matrix{Symbol}
    serialcorr::S
    R::Int64
    σₑ²::Float64
    xmean::Matrix{Float64}
end

# 2. defined undefined class and some rules
struct UndefSNCre <: AbstractUndefSFmodel end
SNCre() = UndefSNCre()
(::UndefSNCre)(args...) = SNCre(args...) 

lagparam(a::AR) = a.p
lagparam(a::MA) = a.q
lagparam(a::ARMA) = a.p + a.q
lagparam(a::SNCre) = lagparam(a.serialcorr)


# 3. bootstrap re-construction rules
# (a::AbstractUndefSFmodel)(selected_row, data::AbstractData)
function (a::SNCre)(selected_row, data::PanelData)
    return SNCre(
        resample(a.dist, selected_row),
        a.ψ,
        a.paramnames,
        a.serialcorr,
        a.R,
        a.σₑ²,
        meanofx(data.rowidx, data.frontiers; verbose = false)
    )

end

# 4. model specific result
# construct the model specific struct
# specify the rule
struct SNCreresult <: AbstractSFresult
    aic::Float64
    bic::Float64
end

function SFresult(main_res::MainSFresult{T, S, U, V}) where{T<:SNCre, S, U, V} 
    aic = -2 * main_res.loglikelihood + numberofparam(main_res.model)
    bic = -2 * main_res.loglikelihood + numberofparam(main_res.model) * log(numberofobs(main_res.data))
    return SFresult(main_res, SNCreresult(aic, bic))
end

sfAIC(a::SFresult) = round(a.model_res.aic, digits = 5)
sfBIC(a::SFresult) = round(a.model_res.bic, digits = 5)

#########################################################################################


#########################################################################################
# TODO: spec(): get the data of parameters, create names for parameters, slicing rules in LLT
# notice that the type should be provide to utilize the multiple dispatch
# spec(model::AbstractSFmodel, df; kwargs...)
#########################################################################################

"""
    meanofx(rowidx, frontiers::Matrix{T}, verbose) where T

To estimate the correlated random effect

# Examples
```juliadoctest
julia> X=[3 5; 4 7; 5 9; 8 2; 3 1]
5×2 Matrix{Int64}:
 3  5
 4  7
 5  9
 8  2
 3  1

julia> rowidx = [1:3, 4:5]


julia> cleanX, pivots = meanofx(rowidx, X); cleanX
* Find Multicollinearity

number 3 column in xmean is dropped
5×2 Matrix{Float64}:
4.0  7.0
4.0  7.0
4.0  7.0
5.5  1.5
5.5  1.5
```
"""
function meanofx(rowidx, frontiers; verbose)
    mean_frontiers = mean.(static_panelize(frontiers, rowidx), dims=1)
    noft = length.(rowidx)
    _xmean = [repeat(i,t) for (i,t) in zip(mean_frontiers, noft)]
    xmean = reduce(vcat, _xmean)
    cons = ones(numberofobs(xmean))

    # if the panel is balanced, the average of variable t is constant
    # `real` ensure the element type is not any which will raise error in `rref_with_pivots`
    # `reduce` for fast `hcat` operation
    xmean, pivots = isMultiCollinearity(:xmean, real(reduce(hcat, (xmean, cons))); warn = verbose)
    return xmean, pivots
end


"""


# Model Specific Arguments
- `ivar`::Union{Symbol, Vector{<:Real}}: further transform to `rowidx` in function `getvar`

- `SCE::Union{AR, MA, ARMA}`: assumption of serial correlation error
- `R::Int`                  : number of correlated random effect simulation
- `σₑ²::Union{Real, Symbol}`: variance of the random error of the correlated random effect

"""
function spec(model::UndefSNCre, df; 
              type::T, dist, σᵥ², ivar, depvar, frontiers, 
              serialcorr, R, σₑ², 
              verbose = true
             ) where{T<:AbstractEconomicType}
    # 1. get some base vaiables
    paneldata, fitted_dist, _col1, _col2 = getvar(
        df, ivar, eval(type), dist, σᵥ², depvar, frontiers, verbose
    )

    # 2. some other variables: σₑ², xmean
    @inbounds σₑ² = σₑ² isa Symbol ? Base.getproperty(df, σₑ²)[1] : σₑ²[1] # since σₑ² will always be constant

    xmean, pivots = meanofx(paneldata.rowidx, paneldata.frontiers, verbose = verbose)
    
    # 3. construct remaind first column of output estimation table
    corrcol1 = isa(serialcorr, AR) ? (:ϕ,) : (isa(serialcorr, MA) ? (:θ,) : (:ϕ, :θ))
    col1     = complete_template(_col1, :αᵢ, :log_σₑ², corrcol1...)

    # 4. construct remaind second column of output estimation tabel
    if !isa(serialcorr, ARMA)
        var = corrcol1[1]
        @inbounds corrcol2 = [[Symbol(var, i) for i = 1:lagparam(serialcorr)]]
    else
        var1, var2 = corrcol1[1], corrcol1[2]
        @inbounds corrcol2 = [
            Symbol[Symbol(var1, i) for i = 1:serialcorr.p], 
            Symbol[Symbol(var2, i) for i = 1:serialcorr.q]
        ]
    end
    xmean_col2 = [Symbol(:mean_, i) for i in _col2[1]]
    push!(xmean_col2, :_cons)

    xmean_col2 = xmean_col2[pivots]
    col2       = complete_template(_col2, xmean_col2, [:_cons], corrcol2...)

    # 5. combine col1 and col2
    paramnames = paramname(col1, col2)

    # 6. generate the rules for slicing parameters
    ψ = complete_template(
        Ψ(paneldata.frontiers, fitted_dist, paneldata.σᵥ²), 
        numberofvar(xmean), 
        1, 
        lagparam(serialcorr)
    )
    push!(ψ, sum(ψ))

    return model(fitted_dist, ψ, paramnames, serialcorr, R, σₑ², xmean), paneldata::PanelData{T}
end


#########################################################################################
# TODO: model specification which will be printed during MLE estimation
# modelinfo(::AbstractUndefSFmodel):
#########################################################################################

(t::AR)(::Symbol) = Symbol(typeof(t), t.p)
(t::MA)(::Symbol) = Symbol(typeof(t), t.q)
function (t::ARMA)(::Symbol)
    str = string(typeof(t))
    return Symbol(str[1:2], t.p, str[3:4], t.q)
end

function (t::AR)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ϕ$i * ϵᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end
function (t::MA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end
function (t::ARMA)(::String)
    ϵᵢₜ = ""
    for i = 1:t.p
        ϵᵢₜ *= "ϕ$i * ϵᵢₜ₋$i  + "
    end
    for i = 1:t.q
        ϵᵢₜ *= "θ$i * ηᵢₜ₋$i  + "
    end
    ϵᵢₜ *= "ηᵢₜ"
    return ϵᵢₜ
end

function modelinfo(model::SNCre)
    modelinfo1 = "flexible panel stochastic frontier model with serially correlated errors"
    
    SCE = model.serialcorr
    modelinfo2 =
"""
    Yᵢₜ = αᵢ + Xᵢₜ*β + T*π + ϵᵢₜ
        where αᵢ = δ₀ + X̄ᵢ'* δ₁ + eᵢ,

        and since the serial correlated assumptions is $(SCE(:type)),
            ϵᵢₜ = $(SCE("info"))
            ηᵢₜ = vᵢₜ - uᵢₜ

        further,     
            vᵢₜ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢₜ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

            eᵢ ∼ N(0, σₑ²)
            σᵤ² = exp(log_σₑ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end

#########################################################################################


# other modules
include("./LLT.jl")
include("extension.jl")
