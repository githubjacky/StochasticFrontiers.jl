function jlmsbc(ξ, model::Cross, data::Data)
    ϵ, σᵥ², dist_param = composite_error(
        slice(ξ, get_paramlength(model), mle=true),
        model, 
        data
    )
    jlms, bc = _jlmsbc(typeofdist(model), σᵥ², dist_param..., ϵ)
   return jlms, bc
end

marginal_data(model::Cross) = _marg_data(model)
marginal_coeff(::Type{Cross{T}}, ξ, ψ) where T = slice(ξ, ψ, mle=true)[2]
unconditional_mean(::Type{Cross{T}}, coeff, args...) where T = _unconditional_mean(T, coeff, args...)