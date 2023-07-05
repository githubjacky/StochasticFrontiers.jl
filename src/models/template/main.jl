#########################################################################################
# TODO: create the type for models
#########################################################################################
# 1. dist, ψ, paramnames are three necessary properties

# 2. defined undefined class and some rules

# 3. bootstrap re-construction rules
# (a::AbstractUndefSFmodel)(selected_row, data::AbstractData)

#########################################################################################








#########################################################################################
# TODO: spec(): get the data of parameters, create names for parameters, slicing rules in LLT
# notice that the type should be provide to utilize the multiple dispatch
# spec(model::AbstractSFmodel, df; kwargs...)
#########################################################################################

    # 1. get some base vaiables

    # 2. some other variables

    # 3. construct remaind first column of output estimation table

    # 4. construct remaind second column of output estimation tabel

    # 5. combine col1 and col2

    # 6. generate the rules for slicing parameters

#########################################################################################











#########################################################################################
# TODO: model specification which will be printed during MLE estimation
# modelinfo(::AbstractUndefSFmodel):
#########################################################################################

#########################################################################################


# other modules
include("./LLT.jl")
include("./extension.jl")
