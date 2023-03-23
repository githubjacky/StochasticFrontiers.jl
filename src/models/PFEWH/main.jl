"""
   PFEWHData(Ỹ::PanelMatrix{<:Real}, X̃::PanelMatrix{<:Real}, hscale::PanelMatrix{<:Real}) <: AbstractModelData

Model Specific data which need to be check for the multicollinearity. It's not
necessay for each model.
"""
struct PFEWHData{T<:PanelMatrix, S<:PanelMatrix, U<:PanelMatrix} <: AbstractModelData 
    Ỹ::T
    X̃::S
    hscale::U
end


"""
    PFEWH(ψ, paramnames, data)

# Arguments
- `ψ::Vector{Any}`: record the length of each parameter, `ψ[end]` is the arrgregate length of all parameters
- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table
- `data::SNCreData`
"""
struct PFEWH{T<:PFEWHData} <: PanelModel
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
    data::T
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
    paneldata, _col1, _col2 = getvar(data, type, ivar, dist, σᵥ², depvar, frontiers)

    # get hscale and demean data 
    hscale_ = Panel(readframe(hscale, df=df), get_rowidx(paneldata))
    Ỹ, X̃ = sf_demean(dependentvar(paneldata)), sf_demean(frontier(paneldata))
    hscale_, _ = isMultiCollinearity(:hscale, hscale_)
   
    # construct remaind first column of output estimation table
    _col1[1] = Symbol(:demean, _col1[1])
    col1 = complete_template(_col1, :hscale)

    # construct remaind second column of output estimation tabel
    col2 = complete_template(_col2, create_names(hscale))

    # costruct the names of parameters of the output estimation table
    paramnames = paramname(col1, col2)

    # generate the remain rule for slicing parameter
    ψ = complete_template(Ψ(paneldata), numberofvar(hscale_))
    push!(ψ, sum(ψ))

    return PFEWH(ψ, paramnames, PFEWHData(Ỹ, X̃, hscale_)), paneldata
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