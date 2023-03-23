function jlmsbc(ξ, model::Cross, data::Data)
    ϵ, σᵥ², dist_param = composite_error(
        slice(ξ, get_paramlength(model), mle=true),
        model, 
        data
    )
    jlms, bc = _jlmsbc(typeofdist(model), σᵥ², dist_param..., ϵ)
   return jlms, bc
end

marginaleffect(ξ, model::Cross, data, bootstrap=false) = _marginaleffect(ξ, model, data, bootstrap)