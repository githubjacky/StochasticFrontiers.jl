"""
    _marg_data(model::SFmodel, args...)

This is the template of getting the marginal effect data and also the number of variables 
for each component.

This template funciton will first select all the distribuiton data. For instance, (μ, σᵤ²) 
in Truncated Normal and (σᵤ²,) in Half Nomral. Then, the data in `args` will be included.

"""
function _marg_data(model::AbstractSFmodel, args...)
    _data = begin
        length(args) == 0 ? 
            [unpack(distof(model))...] : 
            [unpack(distof(model))..., unpack(model, args...)...]
    end
    var_nums = Vector{Int}([numberofvar(i) for i in _data])
    data = reduce(hcat, _data)

    return data, var_nums
end


"""
    _marginal_coeff(::SFmodel, ξ, ψ)

This is the template of getting the coefficients of distribution. Notice that it's not
suitalbe for some models such as those have scaling property.

"""
_marginal_coeff(::Type{<:AbstractSFmodel}, ξ, ψ) = slice(ξ, ψ, mle=true)[2]


"""
    _marginal_label(model, k)

Tis is the template of getting the names of marginal effect, which is exogenous variables
of the distribution, and k is the number of variables of `frontiers`.

"""
function _marginal_label(model, k)
    var_num = sum(numberofvar.(unpack(distof(model))))
    beg_label  = k + 1
    en_label   = beg_label + var_num - 1
    label      = get_paramname(model)[beg_label:en_label, 2]  # use the varmat to get the column name of datafrae

    return label
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
            # println("constant columns: drop $(label)")
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
            # println("duplicated columns: drop $(label)")
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
function marginaleffect(ξ, model, data)
    marg_data, var_nums = marginal_data(model)

    label = marginal_label(model, numberofvar(frontier(data)))

    model_type = typeof(model)
    marg_coeff = marginal_coeff(model_type, ξ, get_paramlength(model))

    mm = similar(marg_data)
    @inbounds for i = axes(mm, 1)
        mm[i, :] = gradient(
            x -> unconditional_mean(model_type, marg_coeff, slice(x, var_nums)...),
            marg_data[i, :]
        )
    end

    mm, label = clean_marginaleffect(mm, label)      # drop the duplicated and constant columns
    considx   = findfirst(x -> x == :_cons, label)   # drop the constant's marginal effect
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
function sfCI(;bootdata=nothing,
               _observed=nothing,
               level=0.05,
               verbose=true
              )
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
    ci = Vector{Vector{float(Int)}}(undef, nofK)
    @inbounds for i = 1:nofK
        data       = bootdata[:,i]
        count      = sum(data .< observed[i])
        z0         = quantile(Normal(), count/nofobs) # bias corrected factor
        alpha1     = cdf(Normal(), z0 + ((z0 + z1) ))
        alpha2     = cdf(Normal(), z0 + ((z0 + z2) ))
        order_data = sort(data)
        cilow      = order_data[Int(ceil(nofobs*alpha1))]
        ciup       = order_data[Int(ceil(nofobs*alpha2))]
        ci[i]      = [round(cilow, digits=5), round(ciup, digits=5)]
    end
    verbose && println("\nBias-Corrected $(100*(1-level))% Confidence Interval:\n")

    return ci
end


"""
    resampling(::Nothing, nobs::Int)
    resampling(rng::AbstractRNG, nobs::Int)

Resampling method through the number of observations. Two difference method base 
on wheter the `AbstractRNG` is given.

"""
resampling(::Nothing, nobs) = sample(1:nobs, nobs; replace=true)
resampling(rng::AbstractRNG, nobs) = sample(rng, 1:nobs, nobs; replace=true)


"""
    bootstrap_marginaleffect(result, R, level, iter, getBootData, seed, verbose)

Main method for bootstrap marginal effect `R` times given the confidence level `level`.
To get the bootstrap data, Set `getBootData` to `true`.

# Arguments
- result::`sfresult`: the return of the function `sfmodel_fit`
- R::`Int`: the number of bootstrap round, default to 500
- level::`AbstractFloat`: confidence level, default to 5%
- iter::`Int`: maximum optimization(MLE) iterations, default to 2000 or user specification in `sfopt`
- getBootData::`Bool`: whether to return the bootstrap data, default to `false`
- seed::`Int`: random seed
- verbose::`Bool`: wheter to print message, default to `true`

"""
function bootstrap_marginaleffect(result;
                                  R::Int      = 500, 
                                  level       = 0.05,
                                  iter::Int   = -1,
                                  getBootData = false,
                                  seed::Int   = -1,
                                  verbose     =true
                                  )
    # check some requirements of data
    0. < level < 1. || throw("The significance level (`level`) should be between 0 and 1.")
    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05

    rng = seed != -1 ? MersenneTwister(seed) : nothing

    maximizer, model, data, options = unpack(result, :ξ, :model, :data, :options)
    iter != -1 && (options[:main_maxit] = iter)
    # options[:warmstart_solver] = nothing

    obs_mm, marg_label = marginaleffect(maximizer, model, data)
    obs_marg_mean      = mean(obs_mm, dims=1)

    p = verbose ? 
        Progress(R, desc = "Resampling: ", color = :white, barlen = 30) :
        Progress(R, desc = "Resampling: ", color = :white, barlen = 30, enabled = false)

    verbose && printstyled(" * bootstrap marginanl effect\n\n", color = :yellow)
    sim_res, i = Matrix{float(Int)}(undef, R, size(obs_marg_mean, 2)), 0
    @inbounds for i = 1:R
        @label start1
        selected_row    = resampling(rng, numberofobs(data))
        bootstrap_data  = data(selected_row)
        bootstrap_model = model(selected_row, bootstrap_data)
        
        Hessian, ξ, _, main_opt = mle(bootstrap_model, bootstrap_data, options, maximizer)

        cond1 = Optim.iteration_limit_reached(main_opt)
        cond2 = isnan(Optim.g_residual(main_opt))  
        cond3 = Optim.g_residual(main_opt) > 1e-1
        (cond1 || cond2 || cond3) && (@goto start1)

        numerical_hessian = hessian!(Hessian, ξ)
        var_cov_matrix = try
            inv(numerical_hessian)
        catch
            @goto start1
        end
        !all(diag(var_cov_matrix) .> 0) && (@goto start1)

        try
            marginal_mean = mean(
                marginaleffect(ξ, bootstrap_model, bootstrap_data)[1], 
                dims=1
            )
            # in this run some of the element is NaN
            sum(isnan.(marginal_mean)) == 0 ? (sim_res[i, :] = marginal_mean) : (@goto start1)
        catch
            @goto start1
        end

        i += 1; next!(p)
    end  # end of the bootstrap process
    theSTD = sqrt.(sum((sim_res .- obs_marg_mean).^2, dims=1) ./ (R-1))
    theSTD = reshape(theSTD, size(theSTD, 2))
    ci_mat = sfCI(bootdata=sim_res, _observed=obs_marg_mean, level=level, verbose=verbose)

    if verbose
        table_content = hcat(marg_label, obs_marg_mean', theSTD, ci_mat)
        header = [
            " "  "mean of the marginal"  "std.err. of the"   "bias-corrected"; 
            " "  "effect on E(u)"        "mean effect"       "$(100*(1-level))%  conf. int.";
        ]
        table = vcat(header, table_content)
        pretty_table(
            table,
            noheader         = true,
            body_hlines      = [2],
            formatters       = ft_printf("%0.5f", 2:4),
            compact_printing = true,
            backend          = Val(result.options[:table_format])
        )
        println()
    end

    res    = hcat(theSTD, ci_mat)
    output = getBootData ? (res, sim_res) : res

    return output
end
