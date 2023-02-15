function jlmsbc(ξ, struc::Cross, data::Data)
    ϵ, σᵥ², dist_param = composite_error(ξ, struc, data)
   return _jlmsbc(typeof(data.dist), σᵥ², dist_param..., ϵ)
end

marginaleffect(ξ, struc::Cross, data, bootstrap=false) = _marginaleffect(ξ, struc, data, bootstrap)