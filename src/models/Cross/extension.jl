#########################################################################################
# TODO: jlmsbc(ξ, model::AbstractSFmodel, data::AbstractData): 
# calculate the jlms, bc index and base equation provided in basic_equations.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

function jlmsbc(ξ, model::Cross, data::Data)
    coeff              = slice(ξ, model.ψ, mle=true)
    ϵ, σᵥ², dist_param = composite_error(coeff, model, data)

    jlms, bc = _jlmsbc(typeofdist(model), σᵥ², dist_param..., ϵ)::NTuple{2, Vector{Float64}}
    return jlms, bc
end

#########################################################################################


#########################################################################################
# TODO: some required funcitons for marginal effect or bootstrap marginal effect    
# some template funcitons are provided in structure/extension.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

# 1. specifiy which portion of data should be used
# marginal_data(model::AbstractSFmodel)
marginal_data(model::Cross) = _marg_data(model)

# 2. specifiy which portion of coefficients should be used
# morginal_coeff(::Type{<:AbstractSFmodel}, ξ, ψ)
marginal_coeff(::Type{<:Cross}, ξ, ψ) = _marginal_coeff(ξ, ψ)

# 3. names for marginal effect 
# marginal_label(model::AbstractSFmodel, k)
marginal_label(model::Cross, k) = _marginal_label(model, k)

# 4. unconditional_mean(::Type{<:AbstractSFmodel}, coeff, args...)
unconditional_mean(::Type{Cross{T}}, coeff, args...) where T = _unconditional_mean(T, coeff, args...)

#########################################################################################
