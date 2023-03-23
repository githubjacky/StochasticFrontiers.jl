function jlmsbc(ξ, model::PFEWH, data::Data)
    ϵ, σᵥ², dist_param = composite_error(
        slice(ξ, get_paramlength(model), mle=true),
        model, 
        data
    )
    jlms, bc = _jlmsbc(typeofdist(data), σᵥ², dist_param..., ϵ)
   return jlms, bc
end

marginaleffect(ξ, model::PFEWH, data, bootstrap=false) = _marginaleffect(ξ, model, data, bootstrap)