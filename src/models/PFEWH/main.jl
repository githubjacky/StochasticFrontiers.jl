"""
   PFEWHData <: AbstractModelData

Model Specific data which need to be check for the multicollinearity. It's not
necessay for each model.
"""
struct PFEWHData <: AbstractModelData end


"""
    PFEWH(ψ, paramnames, data)

# Arguments
- `ψ::Vector{Any}`: record the length of each parameter, `ψ[end]` is the arrgregate length of all parameters
- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table
- `data::SNCreData`
"""
struct PFEWH <: PanelModel
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
    data::PFEWHData
end


# model specification
function sfspec(::Type{Cross}, df...; type, dist, σᵥ², depvar, frontiers)
    # get the vaiables
    crossdata, _col1, _col2 = getvar(df, type, dist, σᵥ², depvar, frontiers)
   
    # construct remaind first column of output estimation table
    col1 = complete_template(_col1, ())

    # construct remaind second column of output estimation tabel
    col2 = complete_template(_col2, ())

    # costruct the names of parameters of the output estimation table
    paramnames = paramname(col1, col2)

    # generate the remain rule for slicing parameter
    ψ = complete_template(Ψ(crossdata), ())
    push!(ψ, sum(ψ))

    return Cross(ψ, paramnames, CrossData()), crossdata
end


function modelinfo(Cross)
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