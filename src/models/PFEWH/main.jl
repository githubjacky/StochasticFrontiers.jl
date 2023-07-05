#########################################################################################
# TODO: create the type for models
#########################################################################################
# 1. dist, ψ, paramnames are three necessary properties
"""
    PFEWH(dist, Ỹ, X̃, hscale, ψ, paramnames)

# Arguments
- `dist::AbstractDist`            : distibution assumption of the inefficiency
-  `Ỹ::AbstractMatrix{<:Real}`    : demean of the dependent variable
- `X̃::AbstractMatrix{<:Real}`     : demean of the explanatory variables
- `hscale::AbstractMatrix{<:Real}`: scaling property

- `ψ::Vector{Any}`                : record the length of each parameter, `ψ[end]` is the
                                  summation of all model parameters' length

- `paramnames::Matrix{Symbol}`    : parameters' names which is the first and second
                                  columns of the output estimation table

"""
struct PFEWH{T<:AbstractDist} <: AbstractPanelModel
    dist::T
    Ỹ::Matrix{Float64}
    X̃::Matrix{Float64}
    hscale::Matrix{Float64}
    ψ::Vector{Int64}
    paramnames::Matrix{Symbol}
end

# 2. defined undefined-class and some rules
struct UndefPFEWH <: AbstractUndefSFmodel end
PFEWH() = UndefPFEWH()
(::UndefPFEWH)(args...) = PFEWH(args...)


# 3. bootstrap re-construction rules
# (a::AbstractUndefSFmodel)(selected_row, data::AbstractData)
function (a::PFEWH)(selected_row, paneldata)
    return PFEWH(
        resample(a.dist, selected_row),
        # depvar, and frontiers has bootstrapped before, no need to select row
        sf_demean(paneldata.depvar, paneldata.rowidx),
        sf_demean(paneldata.frontiers, paneldata.rowidx),
        a.hscale[selected_row, :],
        a.ψ,
        a.paramnames
    )
end

#########################################################################################


#########################################################################################
# TODO: get the data of parameters, create names for parameters, slicing rules in LLT
# notice that the type should be provide to utilize the multiple dispatch
# spec(model::AbstractSFmodel, df; kwargs...)
#########################################################################################

"""


# Model Specific Arguments
- `ivar`::Union{Symbol, Vector{<:Real}}: further transform to `rowidx` in function `getvar`

- `hscale::Union{Symbol, Tuple{Vararg{Symbol}}, Vector{<:Real}, Matrix{<:Real}}`: scaling property

"""
function spec(model::UndefPFEWH, df; 
                type::T, dist, σᵥ², ivar, depvar, frontiers, hscale, verbose = true 
        ) where{T<:AbstractEconomicType}
    # 1. get some base vaiables
    paneldata, dist, _col1, _col2 = getvar(df, ivar, eval(type), dist, σᵥ², depvar, frontiers, verbose)

    # 2. some other variables
    # hscale and demean data 
    h    = isMultiCollinearity(:hscale, readframe(hscale, df=df); warn = verbose)[1]
    dist = isconstant(dist)

    rowidx = paneldata.rowidx
    Ỹ      = sf_demean(paneldata.depvar, rowidx)
    X̃      = sf_demean(paneldata.frontiers, rowidx)
   
    # 3. construct remaind first column of output estimation table
    _col1[1] = Symbol(:demean_, _col1[1])
    col1     = complete_template(_col1, :log_hscale)

    # 4. construct remaind second column of output estimation tabel
    col2 = complete_template(_col2, create_names(hscale))

    # 5. combine col1 and col2
    paramnames = paramname(col1, col2)

    # 6. generate the rules for slicing parameters
    ψ = complete_template(
        Ψ(paneldata.frontiers, dist, paneldata.σᵥ²), 
        numberofvar(h)
    )
    push!(ψ, sum(ψ))
    

    return model(dist, Ỹ, X̃, h, ψ, paramnames), paneldata::PanelData{T}
end

#########################################################################################


#########################################################################################
# TODO: model specification which will be printed during MLE estimation
# modelinfo(::AbstractUndefSFmodel):
#########################################################################################
function modelinfo(::PFEWH)
    modelinfo1 = "panel fixed effect of Wang and Ho (2010 JE)"
    
    modelinfo2 =
"""
    Yᵢₜ = αᵢ + Xᵢₜ*β + ϵᵢₜ
        where ϵᵢₜ = vᵢₜ - uᵢₜ

        further,     
            vᵢₜ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢₜ ∼ hscaleᵢₜ * uᵢ
            hscaleᵢₜ = exp(log_hscaleᵢₜ)

            uᵢ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end

#########################################################################################


# other module
include("./LLT.jl")
include("./extension.jl")
