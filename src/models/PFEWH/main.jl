"""
    PFEWH(fitted_dist, Ỹ, X̃, hscale, ψ, paramnames)

# Arguments
- `fitted_dist::AbstractDist`: distibution assumption of the inefficiency
-  `Ỹ::PanelMatrix{<:Real}`: demean of the dependent variable
- `X̃::PanelMatrix{<:Real}`: demean of the explanatory variables
- `hscale::PanelMatrix{<:Real}`: scaling property
- `ψ::Vector{Any}`: record the length of each parameter, `ψ[end]` is the arrgregate length of all parameters
- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table
"""
struct PFEWH{T<:AbstractDist, 
             S<:PanelMatrix,
             U<:PanelMatrix,
             V<:PanelMatrix
             } <: PanelModel
    fitted_dist::T
    Ỹ::S
    X̃::U
    hscale::V
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
end


# for bootstrap
function (a::PFEWH)(selected_row, paneldata)
    rowidx, depvar, frontiers = unpack(paneldata, (:rowidx, :depvar, :frontiers))
    bootstrap_model = PFEWH(
        typeofdist(a)([i[selected_row, :] for i in unpack(distof(a))]...),
        sf_demean(depvar),
        sf_demean(frontiers),
        Panel(a.hscale[selected_row, :], rowidx),
        get_paramlength(a),
        get_paramname(a)
    )

    return bootstrap_model
end


"""
    sfspec(::Type{Cross}, <arguments>; type, dist, σᵥ², depvar, frontiers)

The model: 

# Arguments
- `data::Union{Tuple{DataFrame}, Tuple{}}`: frame or matrix data
- `type::Union{Type{Production}, Type{Cost}}`: type of economic interpretation
- `dist::Tuple{Union{Half, Trun, Expo}, Vararg{Union{Symbol, Matrix{T}}}} where{T<:Real}`: assumption of the inefficiency
- `σᵥ²::Union{Matrix{T}, Union{Symbol, NTuple{N, Symbol} where N}}`: 
- `ivar::Union{Vector{<:Real}, Sumbol}`: specific data of panel model
- `depvar::Union{AbstractVecOrMat{<:Real}, Symbol}`: dependent variable
- `frontiers::Union{Matrix{<:Real}, NTuple{N, Symbol} where N}`: explanatory variables
- `hscale::Union{AbstractVecOrMat{<:Real}, Symbol}`: scaling property
"""
function sfspec(::Type{PFEWH}, data...; type, dist, σᵥ², ivar, depvar, frontiers, hscale)
    # get the base vaiables
    paneldata, fitted_dist, _col1, _col2 = getvar(data, type, ivar, dist, σᵥ², depvar, frontiers)

    # get hscale and demean data 
    h = Panel(readframe(hscale, df=df), get_rowidx(paneldata))
    h, _ = isMultiCollinearity(:hscale, h)

    Ỹ, X̃ = sf_demean(dependentvar(paneldata)), sf_demean(frontier(paneldata))
    
   
    # construct remaind first column of output estimation table
    _col1[1] = Symbol(:demean, _col1[1])
    col1 = complete_template(_col1, :hscale)

    # construct remaind second column of output estimation tabel
    col2 = complete_template(_col2, create_names(hscale))

    # costruct the names of parameters of the output estimation table
    paramnames = paramname(col1, col2)

    # generate the remain rule for slicing parameter
    ψ = complete_template(
        Ψ(frontier(paneldata), fitted_dist, variance(paneldata)), 
        numberofvar(hscale_)
    )
    push!(ψ, sum(ψ))

    return PFEWH(fitted_dist, ψ, paramnames, PFEWHData(Ỹ, X̃, h)), paneldata
end


function modelinfo(::PFEWH)
    modelinfo1 = "true fixed effect of Wang and Ho (2010 JE)"
    
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


# other module
include("./LLT.jl")
include("./extension.jl")