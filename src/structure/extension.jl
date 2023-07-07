"""
    _marg_data(model::AbstractSFmodel, args...)

This is the template of getting the marginal effect data and also the number of variables 
for each component.

This template funciton will first select all the distribuiton data. For instance, (μ, σᵤ²) 
in Truncated Normal and (σᵤ²,) in Half Nomral. Then, the data in `args` will be included.
One excample for args is the :hscale.

"""
function _marg_data(model::AbstractSFmodel, args...)
    _data = begin
        length(args) == 0 ? 
            Matrix{Float64}[unpack(model.dist)...] : 
            Matrix{Float64}[unpack(model.dist)..., unpack(model, args...)...]
    end
    var_nums = Int64[numberofvar(i) for i in _data]
    data     = reduce(hcat, _data)

    return data, var_nums
end


"""
    _marginal_coeff(::SFmodel, ξ, ψ)

This is the template of getting the coefficients of distribution. Notice that it's not
suitalbe for some models such as those have scaling property.

"""
_marginal_coeff(ξ, ψ) = slice(ξ, ψ, mle=true)[2]


"""
    _marginal_label(model::AbstractSFmodel, k, args...)

Tis is the template of getting the names of marginal effect, which is exogenous variables
of the distribution, and k is the number of variables of `frontiers`. It can be extend by
add the name of variable such as :hscale.

"""
function _marginal_label(model, k, args...)
    num      = length(args)
    label    = Vector{Vector{Symbol}}(undef, num+1)

    var_num  = sum(numberofvar.(unpack(model.dist)))
    beg      = k + 1
    en       = beg + var_num - 1
    label[1] = model.paramnames[beg:en, 2]

    col1     = model.paramnames[:, 1]; push!(col1, :end)
    @inbounds for i = 2:num+1
        beg      = findfirst(x->x==args[i-1], col1)
        en       = findnext(x->x!=Symbol(), col1, beg+1) - 1
        label[i] = model.paramnames[beg:en, 2]
    end

    return reduce(vcat, label)
end


"""
    clean_marginaleffect(m::Matrix{<:Any}, labels::Vector{Symbol})

To drop the constant and duplicated marginal effect
"""
function clean_marginaleffect(m, labels)
    unique_label = unique(labels)
    pos          = Dict([(i, findall(x->x==i, labels)) for i in unique_label])
    id           = Dict([(i, pos[i][1]) for i in unique_label])
    count        = Dict([(i, length(pos[i])) for i in unique_label])
    drop         = []

    @inbounds for (i, label) in enumerate(labels)
        # task1: drop the constant columns
        if length(unique(m[:, i])) == 1
            append!(drop, i)
            count[label] -= 1

            if i == id[label] && count[label] != 0
                id[label] = pos[label][1+(length(pos[label])-count[label])]
            end

            continue
        end
        # task2: drop the columns with duplicated column names
        if i != id[label]
            tar = id[label]
            m[:, tar] = m[:, tar] + m[:, i]
            append!(drop, i)
            count[label] -= 1
        end
    end
    length(labels) == length(drop) && error("there is no marginal effect")

    return m[:, Not(drop)], unique_label
end



"""
    marginaleffect(ξ::Vector{<:Real}, model::SFmodel, data::AbstractData)

Calculate the marginal effect of the unconditional mean in an individual level
"""
function marginaleffect(ξ, model::T, data) where T
    marg_data, var_nums = marginal_data(model)
    label               = marginal_label(model, numberofvar(data.frontiers))
    marg_coeff          = marginal_coeff(T, ξ, model.ψ)
    mm                  = similar(marg_data)

    # because we don't configure the chunk size and rely on ForwardDiff's heuristic
    # we need to pre-allocate the memory to ensure type stability
    out = similar(marg_data, size(marg_data, 2))

    @inbounds for i = axes(mm, 1)
        ForwardDiff.gradient!(
            out,
            x -> unconditional_mean(T, marg_coeff, slice(x, var_nums)...),
            view(marg_data, i, :)
        )
        mm[i, :] .= out
    end

    mm, label = clean_marginaleffect(mm, label)      # drop the duplicated and constant columns
    considx   = findfirst(x -> x == :_cons, label)::Int64   # drop the constant's marginal effect
    mm, label = mm[begin:end, Not(considx)], label[Not(considx)]

    return mm, label
end



"""
    sfCI(; bootdata=nothing, observed=nothing, level=0.5, verbose=false)

Calculate the confidence interval of observed mean marginal effect through bootstrap.

# Arguments
- `bootdata::Matrix{<:Real}`: results of the bootstrap simulation
- `observed::Union{Vector{<:Real}, Real, Tuple, NamedTuple}: observed mean
- `level::AbstractFloat`: confidence level
- `verbose::Bool`

"""
function sfCI(bootdata, _observed; level = 0.05)
    # bias-corrected (but not accelerated) confidence interval 
    # For the "accelerated" factor, need to estimate the SF model 
    # for every jack-knifed sample, which is expensive.
    observed = !isa(_observed ,NamedTuple) ? _observed : values(_observed)
    0. < level < 1. || throw("The significance level (`level`) should be between 0 and 1.")
    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05
    nofobs, nofK = size(bootdata)  # number of statistics
    (nofK == length(observed)) || throw("The number of statistics (`observed`) does not fit the number of columns of bootstrapped data.")

    z1 = quantile(Normal(), level/2)
    z2 = quantile(Normal(), 1 - level/2)  #! why z1 != z2?
    ci = Matrix{Float64}(undef, nofK, 2)

    @inbounds for i = 1:nofK
        data           = bootdata[:,i]
        count          = sum(data .< observed[i])
        z0             = quantile(Normal(), count/nofobs) # bias corrected factor
        alpha1         = cdf(Normal(), z0 + ((z0 + z1) ))
        alpha2         = cdf(Normal(), z0 + ((z0 + z2) ))
        order_data     = sort(data)
        cilow          = order_data[Int(ceil(nofobs*alpha1))]
        ciup           = order_data[Int(ceil(nofobs*alpha2))]
        ci[i, :]      .= [round(cilow, digits=5), round(ciup, digits=5)]
    end


    return ci
end


"""
    resample(::Nothing, nobs::Int)
    resample(rng::AbstractRNG, nobs::Int)

Resampling method through the number of observations. Two difference method base 
on wheter the `AbstractRNG` is given.

"""
resample(::Nothing, nobs)        = Distributions.sample(1:nobs, nobs; replace = true)
resample(rng::AbstractRNG, nobs) = Distributions.sample(rng, 1:nobs, nobs; replace = true)


"""
    bootstrap_marginaleffect(result; R, level, seed, kwargs...)

Main method for bootstrap marginal effect `R` times given the confidence level `level`.
Users can change the options in MLE estimation by specifying same keyword argument in
`sfopt`. Check out  `reset_options`

# Arguments
- result::`sfresult`    : the return of the function `sfmodel_fit`
- R::`Int`              : the number of bootstrap round, default to 500
- level::`AbstractFloat`: confidence level, default to 5%
- seed::`Int`           : random seed

"""
function sfmarginal_bootstrap(result;
                              R::Int64    = 500, 
                              level       = 0.05,
                              seed::Int64 = -1,
                              kwargs...
                              )
    # check some requirements of data
    0. < level < 1. || throw("The significance level (`level`) should be between 0 and 1.")
    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05

    rng = seed != -1 ? MersenneTwister(seed) : nothing

    maximizer, model, data, options = unpack(result.main_res, :ξ, :model, :data, :options)
    options = reset_options(options; kwargs...)

    obs_mm, marg_label = marginaleffect(maximizer, model, data)
    obs_marg_mean      = mean(obs_mm, dims=1)

    p = options.verbose ? 
        Progress(R, desc = "Resampling: ", color = :white, barlen = 30) :
        Progress(R, desc = "Resampling: ", color = :white, barlen = 30, enabled = false)

    sim_res = Matrix{Float64}(undef, R, size(obs_marg_mean, 2))
    iter = 1

    while iter <= R
        selected_row    = resample(rng, numberofobs(data))
        bootstrap_data  = data(selected_row)
        bootstrap_model = model(selected_row, bootstrap_data)
        
        func, ξ, _, main_opt = mle(bootstrap_model, bootstrap_data, options, maximizer)
        Optim.g_converged(main_opt) || continue

        numerical_hessian = hessian!(func, ξ)
        var_cov_matrix = try
            inv(numerical_hessian)
        catch
            continue
        end
        all(diag(var_cov_matrix) .> 0) || continue

        try
            marginal_mean = mean(
                marginaleffect(ξ, bootstrap_model, bootstrap_data)[1], 
                dims=1
            )

            if any(isnan.(marginal_mean)) 
                continue
            else
                @inbounds sim_res[iter, :] = marginal_mean
            end

        catch
            continue
        end

        iter += 1; next!(p)
    end  # end of the bootstrap process

    theSTD = sqrt.(sum((sim_res .- obs_marg_mean).^2, dims=1) ./ (R-1))
    ci_mat = sfCI(sim_res, obs_marg_mean, level = level)

    if options.verbose
        table_content = hcat(marg_label, obs_marg_mean', theSTD', ci_mat)

        l = trunc(Int64, 100 * (1-level))
        header = [
            " ", "mean marginal effect of E(u)", "Std. Err.", "Lower $l%",  "Upper $l%"
        ]

        println("\nBias-Corrected $l% Confidence Interval:\n")

        pretty_table(
            table_content,
            header           = header,
            formatters       = ft_printf("%0.5f", 2:4),
            compact_printing = true,
            backend          = Val(options.table_format)
        )
        println()
    end

    res = hcat(theSTD', ci_mat)

    return res, sim_res
end
