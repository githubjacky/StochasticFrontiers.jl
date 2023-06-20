module StochasticFrontiers
"""
export module functionality
"""

# export functionality for model construction
export 
    usedata, sfspec, sfopt, sfinit, sfmodel_fit, sfmarginal, sfCI, plot_inefficieny,
    sfmaximizer, sfmodel, sfdata, sfoptions, sf_inefficiency, sf_efficiency, sfmaximum
# export general distribution assumption and economic type
export Half, half, Trun, trun, Expo, expo

# export model specific data type
export AR, MA, ARMA  # SNCre


# used packages reference
using Distributions, Random, Statistics, LinearAlgebra, Optim, Plots
import DataFrames: DataFrame
import CSV: File
import DataStructures: OrderedDict
import NLSolversBase: hessian!
import PrettyTables: pretty_table, ft_printf
import HypothesisTests: pvalue
import StatsFuns: normpdf, normcdf, normlogpdf, normlogcdf, log2π
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

# models
include("models/SNCre/main.jl")
include("models/Cross/main.jl")
include("models/PFEWH/main.jl")

include("structure/MLE.jl")
include("structure/main.jl")
include("structure/extension.jl")
include("structure/API.jl")
end  # end of module
