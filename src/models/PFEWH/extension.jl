#########################################################################################
# TODO: jlmsbc(ξ, model::AbstractSFmodel, data::AbstractData): 
# calculate the jlms, bc index and base equation provided in basic_equations.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

function PFEWH_jlmsbc(rowidx, σᵥ², μ, σᵤ², ϵ, h)

    ϵ̃  = demean_panelize(ϵ, rowidx)
    h_ = static_panelize(h, rowidx)
    h̃  = demean_panelize(h, rowidx)

    N    = sum(length.(ϵ̃))
    jlms = zeros(N)
    bc   = zeros(N)

    @floop for (ϵ̃ᵢ, h̃ᵢ, hᵢ, idx) in zip(ϵ̃, h̃, h_, rowidx)
        σₛₛ² = 1.0 / ( h̃ᵢ' * h̃ᵢ * (1/σᵥ²) + 1/σᵤ² )
        σₛₛ  = sqrt(σₛₛ²)
        μₛₛ  = ( μ/σᵤ² - ϵ̃ᵢ' * h̃ᵢ * (1/σᵥ²) ) * σₛₛ²

        jlms[idx] = @. hᵢ * ( μₛₛ + normpdf(μₛₛ/σₛₛ) * σₛₛ / normcdf(μₛₛ/σₛₛ) )
        bc[idx]   = @. ( (normcdf(μₛₛ/σₛₛ-hᵢ*σₛₛ)) / normcdf(μₛₛ/σₛₛ) ) * 
                       exp( -hᵢ * μₛₛ + 0.5 * (hᵢ^2) * σₛₛ² )
    end

    return jlms, bc
end


function jlmsbc(ξ, model::PFEWH, data::PanelData)
    ϵ, σᵥ², dist_param, h = composite_error(
        slice(ξ, model.ψ, mle=true),
        model, 
        data
    )
    jlms, bc = begin
        model.dist isa Trun ?
        PFEWH_jlmsbc(data.rowidx, σᵥ², dist_param..., ϵ, h) :
        PFEWH_jlmsbc(data.rowidx, σᵥ², 0., dist_param..., ϵ, h)
    end

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
marginal_data(model::PFEWH) = _marg_data(model, :hscale)

# 2. specifiy which portion of coefficients should be used
# morginal_coeff(::Type{<:AbstractSFmodel}, ξ, ψ)
marginal_coeff(::Type{<:PFEWH}, ξ, ψ) = slice(ξ, ψ, mle=true)[[2, 4]]

# 3. names for marginal effect 
# marginal_label(model::AbstractSFmodel, k)
marginal_label(model::PFEWH, k) = _marginal_label(model, k, :log_hscale)

# 4. unconditional_mean(::Type{<:AbstractSFmodel}, coeff, args...)
function PFEWH_unconditional_mean(coeff, μ, log_σᵤ², _log_h)

    Wμ = μ == zeros(1) ? zeros(1) : coeff[1][begin:begin+length(μ)-1]
    Wᵤ = coeff[1][end-length(log_σᵤ²)+1:end]

    log_h = _log_h' * coeff[2]

    μ       = exp(log_h) * (μ' * Wμ)
    σᵤ      = exp(log_h + 0.5 * log_σᵤ²' * Wᵤ)
    Λ       = μ / σᵤ
    uncondU = σᵤ * ( Λ + normpdf(Λ) / normcdf(Λ) )

    return uncondU
end

function unconditional_mean(::Type{PFEWH{Half{T}}}, coeff, log_σᵤ², log_h) where T
    return PFEWH_unconditional_mean(coeff, [0.], log_σᵤ², log_h)
end

function unconditional_mean(::Type{PFEWH{Trun{T, S}}}, coeff, μ, log_σᵤ², log_h) where{T, S}
    return PFEWH_unconditional_mean(coeff, μ, log_σᵤ², log_h)
end
