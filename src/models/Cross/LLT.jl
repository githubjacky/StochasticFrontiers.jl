#########################################################################################
# TODO: composite error term
# composite_error(coeff::Vector{Vector{T}}, model::AbstractModel, data) where T
#########################################################################################

function composite_error(coeff::Vector{Vector{T}}, model::Cross, data) where T
    σᵥ²        = exp.(data.σᵥ² * coeff[3])
    dist_param = model.dist(coeff[2])
    ϵ          = (data.econtype * (data.depvar- data.frontiers*coeff[1]))[:, 1]
    
    return ϵ, σᵥ², dist_param
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

function LLT(ξ, model::Cross, data::Data)
    coeff              = slice(ξ, model.ψ, mle=true)
    ϵ, σᵥ², dist_param = composite_error(coeff, model, data)

    return _loglikelihood(typeofdist(model), σᵥ², dist_param..., ϵ)
end

#########################################################################################
