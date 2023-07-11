#########################################################################################
# TODO: create the type for models
#########################################################################################
# 1. dist, ψ, paramnames are three necessary properties
"""
    Cross(dist, ψ, paramnames)

Notice: these threee properties must be included in every model type.

# Arguments
- `dist::AbstractDist`        : distibution assumption of the inefficiency

- `ψ::Vector{Int64}`          : record the length of each parameter, `ψ[end]` is the 
                              summation of of all parameters' length.

- `paramnames::Matrix{Symbol}`: parameters' names used by the output estimation table

"""
struct Cross{T<:AbstractDist} <: AbstractSFmodel
    dist::T
    ψ::Vector{Int64}
    paramnames::Matrix{Symbol}
end

# 2. defined undefined class and some rules
struct UndefCross <: AbstractUndefSFmodel end
Cross() = UndefCross()
(::UndefCross)(args...) = Cross(args...) 



# 3. bootstrap re-construction rules
# (a::AbstractUndefSFmodel)(selected_row, data::AbstractData)
function (a::Cross)(selected_row, ::Data)
    bootstrap_model = Cross(
        resample(a.dist, selected_row),
        a.ψ,
        a.paramnames
    )

    return bootstrap_model
end


# 4. model specific result
# construct the model specific struct
# specify the rule
struct Crossresult <: AbstractSFresult end

function SFresult(main_res::MainSFresult{T, S, U, V}) where{T<:Cross, S, U, V}
    return SFresult(main_res, Crossresult())
end

#########################################################################################


#########################################################################################
# TODO: spec: get the data of parameters, create names for parameters, slicing rules in LLT
# notice that the type should be provide to utilize the multiple dispatch
# spec(model::AbstractSFmodel, df; kwargs...)
#########################################################################################
"""
# Base Arguments
- `model::Type{<:AbstractUndefSFmodel}`: undefined model to ensure type stability.

- `df::Union{DataFrame, Nothing}`: `df isa Nothing` is the case when inputs are matrix data.

- `type::Union{Type{Production}, Type{Cost}}`: type of economic interpretation

- `dist::Tuple{Union{Half, Trun, Expo}, Vararg{Union{Symbol, Matrix{T}}}} where{T<:Real}`: 
   distribution assumption of the inefficiency

- `σᵥ²::Union{Matrix{T}, Union{Symbol, NTuple{N, Symbol} where N}}`

- `depvar::Union{AbstractVecOrMat{<:Real}, Symbol}`            : dependent variable
- `frontiers::Union{Matrix{<:Real}, NTuple{N, Symbol} where N}`: explanatory variables

"""
function spec(model::UndefCross, df; 
              type::T, dist, σᵥ², depvar, frontiers, 
              verbose = true
             ) where{T<:AbstractEconomicType}
    # 1. get some base vaiables
    crossdata, dist, _col1, _col2 = getvar(
        df, type, dist, σᵥ², depvar, frontiers, verbose
    )

    # 2. some other variables

    # 3. construct remaind first column of output estimation table
    col1 = complete_template(_col1)

    # 4. construct remaind second column of output estimation tabel
    col2 = complete_template(_col2)

    # 5. combine col1 and col2
    paramnames = paramname(col1, col2)

    # 6. generate the rules for slicing parameters
    ψ = complete_template(
        Ψ(crossdata.frontiers, dist, crossdata.σᵥ²)
    )
    push!(ψ, sum(ψ))
    
    return model(dist, ψ, paramnames), crossdata::Data{T}
end

#########################################################################################


#########################################################################################
# TODO: modelinfo(): model specification which will be printed during MLE estimation
#########################################################################################
function modelinfo(::Cross)
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

#########################################################################################


# other module
include("./LLT.jl")
include("./extension.jl")
