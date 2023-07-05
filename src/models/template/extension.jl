#########################################################################################
# TODO: jlmsbc(ξ, model::AbstractSFmodel, data::AbstractData): 
# calculate the jlms, bc index and base equation provided in basic_equations.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

#########################################################################################











#########################################################################################
# TODO: some required funcitons for marginal effect or bootstrap marginal effect    
# some template funcitons are provided in structure/extension.jl
# notice that the type should be provide to utilize the multiple dispatch
#########################################################################################

# 1. specifiy which portion of data should be used
# marginal_data(model::AbstractSFmodel)

# 2. specifiy which portion of coefficients should be used
# morginal_coeff(::Type{<:AbstractSFmodel}, ξ, ψ)

# 3. names for marginal effect 
# marginal_label(model::AbstractSFmodel, k)

# 4. unconditional_mean(::Type{<:AbstractSFmodel}, coeff, args...)

#########################################################################################
