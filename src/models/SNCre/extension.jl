#########################################################################################
# TODO: jlmsbc(ξ, model::AbstractSFmodel, data::AbstractData): 
# calculate the jlms, bc index and base equation provided in basic_equations.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

"""
     check_idgood(simulate_η::Panelized{T}) where T 

Check whether each simulation column has infinite value for all individual and 
drop the column if there exist infinite value

# Examples
```juliadoctest
julia> a = [1 2 4; 3 Inf 9]
2×3 Matrix{Float64}:
 1.0   2.0  4.0
 3.0  Inf   9.0

julia> check_idgood(a)
2×2 Matrix{Float64}:
 1.0  4.0
 3.0  9.0

```

"""
function check_idgood(simulate_η)
    infcol_ptr = map(
        x -> any(isinf.(x)), 
        eachcol(simulate_η)
    )
    infcol_idx = findall(x->x==1, infcol_ptr)

    return simulate_η[:, Not(infcol_idx)]
end

function jlmsbc(ξ, model::SNCre, data::PanelData)

    coeff                       = slice(ξ, model.ψ, mle=true) 
    simulate_η, σᵥ², dist_param = composite_error(coeff, model, data)

    σᵥ²_ = lagdrop(σᵥ², data.rowidx, lagparam(model))

    dist_param_ = map(
        x -> lagdrop(x, data.rowidx, lagparam(model)),
        dist_param
    )

    η = view(mean(check_idgood(simulate_η), dims = 2), :, 1)

    jlms, bc = _jlmsbc(typeofdist(model), σᵥ²_, dist_param_..., η)::NTuple{2, Vector{Float64}}

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
marginal_data(model::SNCre) = _marg_data(model)

# 2. specifiy which portion of coefficients should be used
# morginal_coeff(::Type{<:AbstractSFmodel}, ξ, ψ)
margianl_coeff(::Type{<:SNCre}, ξ, ψ) = _marginal_coeff(ξ, ψ)

# 3. names for marginal effect 
# marginal_label(model::AbstractSFmodel, k)
marginal_label(model::SNCre, k) = _marginal_label(model, k)

# 4. unconditional_mean(::Type{<:AbstractSFmodel}, coeff, args...)
unconditional_mean(::Type{SNCre{T, S}}, coeff, args...) where{T, S}  = _unconditional_mean(T, coeff, args...)

#########################################################################################
