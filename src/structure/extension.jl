"""
    sfmarginal(result::NamedTuple; bootstrap=false, kwargs...)

The primary interface for usage, and the main input is the output `result`
from the mle estimation.
"""
function sfmarginal(result; bootstrap=false, kwargs...)
    if !bootstrap
        marginaleffect(result.ξ, result.model, result.data, bootstrap)
    else
        bootstrap_marginaleffect(result; kwargs...)
    end
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

function sfCI(
    ;bootdata=nothing,
    _observed=nothing,
    level=0.05,
    verbose=false
    )
    # bias-corrected (but not accelerated) confidence interval 
    # For the "accelerated" factor, need to estimate the SF model 
    # for every jack-knifed sample, which is expensive.
    observed = !isa(_observed ,NamedTuple) ? _observed : values(_observed)
    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")
    level > 0.50 && (level = 1-level)  # 0.95 -> 0.05
    nofobs, nofK = size(bootdata)  # number of statistics
    (nofK == length(observed)) || throw("The number of statistics (`observed`) does not fit the number of columns of bootstrapped data.")

    z1 = quantile(Normal(), level/2)
    z2 = quantile(Normal(), 1 - level/2)  #! why z1 != z2?
    ci = Vector{NTuple{2, float(Int)}}(undef, nofK)
    @inbounds for i in 1:nofK
        data = bootdata[:,i]
        count = sum(data .< observed[i])
        z0 = quantile(Normal(), count/nofobs) # bias corrected factor
        alpha1 = cdf(Normal(), z0 + ((z0 + z1) ))
        alpha2 = cdf(Normal(), z0 + ((z0 + z2) ))
        order_data = sort(data)
        cilow = order_data[Int(ceil(nofobs*alpha1))]
        ciup = order_data[Int(ceil(nofobs*alpha2))]
        ci[i] = (round(cilow, digits=5), round(ciup, digits=5))
    end
    verbose && println("\nBias-Corrected $(100*(1-level))% Confidence Interval:\n")

    return ci
end


"""
    bootstrap_marginaleffect(result, mymisc, R, level, iter, getBootData, seed, verbose)

Main method for bootstrap marginal effect
- 
"""
function bootstrap_marginaleffect(
    result::sfresult;
    mymisc=nothing,
    R::Int=500, 
    level=0.05,
    iter::Int=-1,
    getBootData=false,
    seed::Int=-1,
    verbose=true)
    # check some requirements of data
    ((level > 0.0) && (level < 1.0)) || throw("The significance level (`level`) should be between 0 and 1.")

    level > 0.5 && (level = 1-level)  # 0.95 -> 0.05

    # In the following lines, the integer part had been taken care of in Type.
    (seed == -1) || ( seed > 0) || throw("`seed` needs to be a positive integer.")
    (iter == -1) || ( iter > 0) || throw("`iter` needs to be a positive integer.")
    (R > 0) || throw("`R` needs to be a positive integer.")

    maximizer, model, data, options = unpack(result, (:ξ, :model, :data, :options))
    iter > 0 && (options[:main_maxit] = iter)

    _, obs_marg_mean = marginaleffect(maximizer, model, data)
    rng = seed != -1 ? MersenneTwister(seed) : -1
    options[:warmstart_solver] = nothing
    p = Progress(R, desc="Sampling: ", color=:white, barlen=30)

    printstyled(" * bootstrap marginanl effect\n\n", color=:yellow)
    sim_res = Matrix{Real}(undef, R, length(obs_marg_mean))
    @inbounds for i in 1:R
        @label start1
        if rng != -1
            selected_row = sample(rng, 1:numberofobs(data), numberofobs(data); replace=true)
        else
            selected_row = sample(1:numberofobs(data), numberofobs(data); replace=true)
        end
        bootstrap_data = data(selected_row)
        bootstrap_model = model(selected_row, bootstrap_data)
        
        Hessian, ξ, _, main_opt = mle(bootstrap_model, bootstrap_data, options, maximizer)
        if Optim.iteration_limit_reached(main_opt) || 
           isnan(Optim.g_residual(main_opt)) ||  
           Optim.g_residual(main_opt) > 1e-1
               @goto start1
        end
        numerical_hessian = hessian!(Hessian, ξ)
        var_cov_matrix = try
            inv(numerical_hessian)
        catch
            @goto start1
        end
        !all(diag(var_cov_matrix) .> 0) && (@goto start1)

        _, marginal_mean = marginaleffect(ξ, bootstrap_model, bootstrap_data, true)
        # in this run some of the element is NaN
        sum(isnan.(marginal_mean)) == 0 ? (sim_res[i, :] = marginal_mean) : (@goto start1)
        
        next!(p)
    end  # for i=1:R
    
    theMean = collect(values(obs_marg_mean))
    theSTD = sqrt.(sum((sim_res .- theMean').^2, dims=1) ./(R-1))
    theSTD = reshape(theSTD, size(theSTD, 2))
    ci_mat = sfCI(bootdata=sim_res, _observed=theMean, level=level, verbose=verbose)

    if verbose
        table_content = hcat(collect(keys(obs_marg_mean)), theMean, theSTD, ci_mat)
        table = [" " "mean of the marginal" "std.err. of the"  "bias-corrected"; 
                 " " "effect on E(u)"       "mean effect"      "$(100*(1-level))%  conf. int.";
                 table_content]
        pretty_table(
            table,
            noheader=true,
            body_hlines = [2],
            formatters = ft_printf("%0.5f", 2:4),
            compact_printing = true,
            backend = Val(result.options[:table_format])
        )
        println()
    end
    getBootData ? (return hcat(theSTD, ci_mat), sim_res) : (return hcat(theSTD, ci_mat))
end