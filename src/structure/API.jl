
"""
    sfmarginal(result::NamedTuple; bootstrap=false, kwargs...)

By default, calculate the observational marginal effects. Set `bootstrap` to true, if you
want to bootstrap the mean marginal effects. Check the function `bootstrap_marginaleffect`
to see the keyword arguments for bootstrap operation.

"""
function sfmarginal(result; bootstrap=false, kwargs...)
    if !bootstrap
        mm, label = marginaleffect(result.ξ, result.model, result.data)
        label = [Symbol(:marg_, i) for i in label]
        return DataFrame(mm, label), NamedTuple{Tuple(label)}(mean(mm, dims=1))
    else
        bootstrap_marginaleffect(result; kwargs...)
    end
end


# output the estimation result
struct sfresult{T, S}
    ξ::Vector{Float64}
    model::T
    data::S
    options::OrderedDict{Symbol, Any}
    jlms::Vector{Float64}
    bc::Vector{Float64}
    loglikelihood::Float64
end


sfmaximizer(a::sfresult)     = getproperty(a, :ξ)
sfmodel(a::sfresult)         = getproperty(a, :model)
sfdata(a::sfresult)          = getproperty(a, :data)
sfoptions(a::sfresult)       = getproperty(a, :options)
sf_inefficiency(a::sfresult) = getproperty(a, :jlms)
sf_efficiency(a::sfresult)   = getproperty(a, :bc)
sfmaximum(a::sfresult)       = getproperty(a, :loglikelihood)


"""
    plot_inefficieny(res::sfresult)

Plot the inefficiency and efficiency index

"""
function plot_inefficieny(res::sfresult)
    plot(
        histogram(sf_inefficiency(res), xlabel="JLMS", bins=100, label=""),
        histogram(sf_efficiency(res), xlabel="BC", bins=50, label=""),
        layout = (1,2), legend=false
    )
end
