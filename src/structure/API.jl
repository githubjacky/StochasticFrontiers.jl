"""
    sfmarginal(result::NamedTuple; bootstrap=false, kwargs...)

By default, calculate the observational marginal effects. Set `bootstrap` to true, if you
want to bootstrap the mean marginal effects. Check the function `bootstrap_marginaleffect`
to see the keyword arguments for bootstrap operation.

"""
function sfmarginal(result::SFresult)
    mm, label     = marginaleffect(result.main_res.ξ, result.main_res.model, result.main_res.data)
    label         = Symbol[Symbol(:marg_, i) for i in label]

    key = Tuple(label)
    val = mean(mm, dims=1)

    return DataFrame(mm, label), NamedTuple{key}(val)
end


sfmaximizer(a::SFresult)     = a.main_res.ξ
sfmodel(a::SFresult)         = a.main_res.model
sfdata(a::SFresult)          = a.main_res.data
sfoptions(a::SFresult)       = a.main_res.options
sfstartpt(a::SFresult)       = a.main_res.startpt
sf_inefficiency(a::SFresult) = a.main_res.jlms
sf_efficiency(a::SFresult)   = a.main_res.bc
sfmaximum(a::SFresult)       = a.main_res.loglikelihood
sftrace(a::SFresult)         = Optim.g_norm_trace(a.main_res.fit_res)
sfcheck_converge(a::SFresult) = println(a.main_res.fit_res)


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
