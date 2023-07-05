#########################################################################################
# TODO: composite error term
# composite_error(coeff::Vector{Vector{T}}, model::AbstractModel, data) where T
#########################################################################################

function composite_error(coeff, model::PFEWH, data)
    σᵥ²        = exp( (data.σᵥ² * coeff[3])[1] )  # σᵥ² is a scalar
    dist_param = [ i[1] for i in model.dist(coeff[2]) ]  # all dist_param are scalars
    h          = exp.( model.hscale * coeff[4] )

    ϵ = ( data.econtype * (model.Ỹ - model.X̃*coeff[1]) )[:, 1]

    return ϵ, σᵥ², dist_param, h
end

#########################################################################################


#########################################################################################
# TODO: loglikelihood  function
# template functions are provided if needed
# notice that the type should be provide to utilize the multiple dispatch
# there is no need to reverse the sign because the minus symbol(-) will be added
# in structure/mle.jl
# LLT(ξ, model::AbstractModel, data::Data)
#########################################################################################

function PFEWH_llh(rowidx, σᵥ², μ, σᵤ², ϵ, h, γ, δ2)
    ϵ̃  = demean_panelize(ϵ, rowidx)
    h̃  = demean_panelize(h, rowidx)
    T  = length.(ϵ̃)

    @floop for (ϵ̃ᵢ, h̃ᵢ, Tᵢ) in zip(ϵ̃, h̃, T)
        σₛₛ² = 1 / ( h̃ᵢ' * h̃ᵢ * (1/σᵥ²) + (1/σᵤ²) )
        μₛₛ  = ( μ/σᵤ² - (ϵ̃ᵢ' * h̃ᵢ * (1/σᵥ²))[1] ) * σₛₛ²
        
        es2  = -0.5 * sum(ϵ̃ᵢ.^2) * (1/σᵥ²)
        KK   = -0.5 * (Tᵢ-1) * log2π - 0.5 * (Tᵢ-1) * γ
        llhᵢ = KK + es2 +
               0.5 * ( μₛₛ^2/σₛₛ² - μ^2/σᵤ² ) +
               0.5 * log(σₛₛ²) + 
               normlogcdf( μₛₛ/sqrt(σₛₛ²) ) -
               0.5 * δ2 - 
               normlogcdf( μ/sqrt(σᵤ²) )

        @reduce llh = 0. + llhᵢ
    end

    return llh
end


function LLT(ξ, model::PFEWH{<:Trun}, data::PanelData)
    coeff                  = slice(ξ, model.ψ, mle = true)
    ϵ, σᵥ², dist_param, h = composite_error(coeff, model, data)

    llh = PFEWH_llh(data.rowidx, σᵥ², dist_param..., ϵ, h, coeff[3][1], coeff[2][2])

    return llh
end


function LLT(ξ, model::PFEWH{<:Half}, data::PanelData)
    coeff                  = slice(ξ, model.ψ, mle = true)
    ϵ, σᵥ², dist_param, h = composite_error(coeff, model, data)
    
    llh = PFEWH_llh(data.rowidx, σᵥ², 0., dist_param..., ϵ, h, coeff[3][1], coeff[2][1])

    return llh
end
