"""
   CrossData <: AbstractModelData

Model Specific data which need to be check for the multicollinearity. It's not
necessay for each model.
"""
struct CrossData <: AbstractModelData end


"""
    Cross(ψ, paramnames, data)

# Arguments
- `ψ::Vector{Any}`: record the length of each parameter, `ψ[end]` is the arrgregate length of all parameters
- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table
- `data::CrossData`
"""
struct Cross <: SFmodel
    ψ::Vector{Any}
    paramnames::Matrix{Symbol}
    data::CrossData
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
- `SCE::Union{AR, MA, ARMA}`: assumption of serial correlation
- `R::Int`: number of correlated random effect simulation
- `σₑ²::Union{Real, Symbol}`: variance of the random error of the correlated random effect
"""
function sfspec(::Type{Cross}, data...; type, dist, σᵥ², depvar, frontiers)
    # get the vaiables
    crossdata, _col1, _col2 = getvar(data, type, dist, σᵥ², depvar, frontiers)
   
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
    Yᵢ = Xᵢ*β + + ϵᵢ
        where ϵᵢ = vᵢ - uᵢ

        further,     
            vᵢ ∼ N(0, σᵥ²),
            σᵥ²  = exp(log_σᵥ²)

            uᵢ ∼ N⁺(0, σᵤ²),
            σᵤ² = exp(log_σᵤ²)

    In the case of type(cost), "- uᵢₜ" above should be changed to "+ uᵢₜ"
"""
    _modelinfo(modelinfo1, modelinfo2)
end


# other module
include("./LLT.jl")
include("./extension.jl")