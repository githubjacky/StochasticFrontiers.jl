struct CrossData <: AbstractModelData end

struct Cross <: SFmodel
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
    data::CrossData
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
    modelinfo1 = "Base stochastic frontier model"
    
    modelinfo2 =
"""
    Yᵢₜ = Xᵢₜ*β + + ϵᵢₜ
        where ϵᵢₜ = vᵢₜ - uᵢₜ

        further,     
            vᵢₜ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢₜ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end


# other module
include("./LLT.jl")
include("./extension.jl")