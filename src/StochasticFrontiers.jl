module StochasticFrontiers

# export functionality for model construction
export sfspec, sfopt, sfinit, sfmodel_fit, sfmarginal, sfCI, sfmarginal_bootstrap

# api for  `sfresult`
export plot_inefficieny, sfmaximizer, sfmodel, sfdata, sfstartpt, sfoptions, sf_inefficiency, 
       sf_efficiency, sfmaximum, sfcheck_converge, sftrace, sfAIC, sfBIC

# model type
export Cross, PFEWH, SNCre

# Economics type
export Prod, prod, p, Cost, cost

# export general distribution assumption and economic type
export Half, half, Trun, trun, Expo, expo

# export model specific data type
export AR, MA, ARMA  # SNCre

export Optim

# used packages reference
using Distributions, Random, Statistics, LinearAlgebra, Optim, Plots, StaticArrays,
      DataFrames, CSV, DataStructures, NLSolversBase, PrettyTables, HypothesisTests,
      StatsFuns, StatsBase, Polynomials, RowEchelon, ForwardDiff, InvertedIndices,
      ProgressMeter, FLoops, Optim

import Base: convert


# Modulize the source code
include("types.jl")
include("utils.jl")
include("basic_equations.jl")

include("structure/MLE.jl")
include("structure/main.jl")

# models
include("models/Cross/main.jl")
include("models/PFEWH/main.jl")
include("models/SNCre/main.jl")

include("structure/extension.jl")
include("structure/api.jl")
end  # end of module
