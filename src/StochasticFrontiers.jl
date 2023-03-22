module StochasticFrontiers
"""
export module functionality
"""
# export model
export Cross, SNCre

# export functionality for model construction
export usedata, sfspec, sfopt, sfinit, sfmodel_fit, sfmarginal, sfCI

# export general distribution assumption and economic type
export Half, half, h, Trun, trun, t, Expo, expo, e
export Prod, prod, p, Cost, cost

# export model specific data type
export AR, MA, ARMA  # SNCre

# export package-Optim's algorithms
export NelderMead, SimulatedAnnealing, SAMIN, ParticleSwarm,
       ConjugateGradient, GradientDescent, BFGS, LBFGS,
       Newton, NewtonTrustRegion, IPNewton


# used packages reference
using Distributions, Random, Statistics, LinearAlgebra, Optim
import DataFrames: DataFrame
import CSV: File
import DataStructures: OrderedDict
import NLSolversBase: hessian!
import PrettyTables: pretty_table, ft_printf
import HypothesisTests: pvalue
import StatsFuns: normpdf, normcdf, normlogpdf, normlogcdf
import Polynomials: fromroots, coeffs
import RowEchelon: rref_with_pivots
import ForwardDiff: gradient
import InvertedIndices: Not
import ProgressMeter: Progress, BarGlyphs, next!
import LoopVectorization: @tturbo
import FLoops: @floop, @reduce

import Optim as opt  # for better understanding of the origin of the method



# Modulize the source code
include("types.jl")
include("utils.jl")
include("function.jl")

include("models/SNCre/main.jl")
include("models/Cross/main.jl")
include("models/PFEWH/main.jl")

include("structure/MLE.jl")
include("structure/main.jl")
include("structure/extension.jl")
end  # end of module --SFrontiers