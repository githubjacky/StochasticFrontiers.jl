
"""
    sfmarginal(result::NamedTuple; bootstrap=false, kwargs...)

By default, calculate the observational marginal effects. Set `bootstrap` to true, if you
want to bootstrap the mean marginal effects. Check the function `bootstrap_marginaleffect`
to see the keyword arguments for bootstrap operation.

"""
function sfmarginal(result; verbose = true)
    mm, label     = marginaleffect(result.ξ, result.model, result.data)
    label         = Symbol[Symbol(:marg_, i) for i in label]

    key = Tuple(label)
    val = mean(mm, dims=1)

    return DataFrame(mm, label), NamedTuple{key}(val)
end


# output the estimation result
struct SFresult{T, S, U, V}
    ξ::Vector{Float64}
    model::T
    data::S
    options::U
    jlms::Vector{Float64}
    bc::Vector{Float64}
    loglikelihood::Float64
    main_opt::V
end


sfmaximizer(a::SFresult)     = a.ξ
sfmodel(a::SFresult)         = a.model
sfdata(a::SFresult)          = a.data
sfoptions(a::SFresult)       = a.options
sf_inefficiency(a::SFresult) = a.jlms
sf_efficiency(a::SFresult)   = a.bc
sfmaximum(a::SFresult)       = a.loglikelihood
sftrace(a::SFresult)         = Optim.g_norm_trace(a.main_opt)
sfcheck_converge(a::SFresult) = println(a.main_opt)


"""
Plot the inefficiency and efficiency index
"""
function plot_inefficieny(res::SFresult)
    plot(
        histogram(sf_inefficiency(res), xlabel="JLMS", bins=100, label=""),
        histogram(sf_efficiency(res), xlabel="BC", bins=50, label=""),
        layout = (1,2), legend=false
    )
end
