"""
     check_idgood(simulate_η::Panelized{T}) where T 

Check whether each simulation column has infinite value for all individual and 
drop the column if there exist infinite value

# Examples
```juliadoctest
julia> simulate_η = Panelized(
           [
               [1 2; 
               3 Inf],
               [5 6; 
                7 8]
            ], 
            [1:2, 3:4]
       )
2-element Main.SFrontiers.Panelized{Vector{Matrix{Float64}}}:
 [1.0 2.0; 3.0 Inf]
 [5.0 6.0; 7.0 8.0]

julia> a = check_idgood(simulate_η)
4-element Main.SFrontiers.PanelVector{Float64}:
 1.0
 3.0
 5.5
 7.5
```
"""
function check_idgood(simulate_η)
    _η = [
        mean(
            i[:, (sum(isinf.(i), dims=1)[1, :] .== 0)],
            dims=2
        )[:, 1]
        for i in simulate_η
    ]
    η = Panel(reduce(vcat, _η), simulate_η.rowidx)
    return η
end

function jlmsbc(ξ, struc::SNCre, data::PanelData)
    simulate_η, σᵥ², dist_param = composite_error(ξ, struc, data)

    η = check_idgood(simulate_η)

    jlms, bc = _jlmsbc(
        typeof(data.dist), Panel(σᵥ²), [Panel(i) for i in dist_param]..., η
    )
   return jlms, bc
end

marginaleffect(ξ, model::SNCre{T}, data, bootstrap=false) where T = _marginaleffect(ξ, model, data, bootstrap)