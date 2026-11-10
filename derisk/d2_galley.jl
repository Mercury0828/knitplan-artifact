# D2 — run Galley on the exported reconstruction queries (the pre-registered protocol §D2).
# Usage: julia --project=D:\runs\knitplan\galley_env d2_galley.jl <case> [<case> ...]
# For each case: build Finch tensors from the exported arrays (sparse: SparseList levels; dense: Dense levels),
# run galley() once cold (includes Julia compilation) and 5 times warm; record opt_time / execute_time / wall time,
# the logical plan text, and write the answer for the equivalence check.
using Galley, Galley.Optimizer, NPZ, JSON

const DIR = get(ENV, "GALLEY_DIR", raw"D:\runs\knitplan\d2")  # E1-G sets GALLEY_DIR=D:\runs\knitplan\e1\galley

function mk(arr, fmt)
    n = ndims(arr)
    lvl = Element(0.0)
    for _ in 1:n
        lvl = fmt == "dense" ? Dense(lvl) : SparseList(lvl)
    end
    return Tensor(lvl, arr)
end

function run_case(case)
    spec = JSON.parsefile(joinpath(DIR, case * ".json"))
    data = npzread(joinpath(DIR, case * ".npz"))
    inputs = []
    for t in spec["tensors"]
        arr = Array{Float64}(data[t["key"]])
        if ndims(arr) == 0
            arr = reshape(arr, 1)
        end
        push!(inputs, Input(mk(arr, t["fmt"]), Symbol.(t["idx"])...))
    end
    outs = Symbol.(spec["outputs"])
    allidx = unique(vcat([Symbol.(t["idx"]) for t in spec["tensors"]]...))
    summed = [i for i in allidx if !(i in outs)]
    expr = MapJoin(*, inputs...)
    body = isempty(summed) ? expr : Agg(+, 0.0, summed..., expr)  # Aggregate(op, init, idxs..., arg)
    q = isempty(outs) ? Mat(body) : Mat(outs..., body)
    res = Dict{String,Any}("case" => case)
    t0 = time(); r = galley(q; verbose=0); res["cold_wall"] = time() - t0
    res["cold_opt"] = r.opt_time; res["cold_exec"] = r.execute_time
    walls = Float64[]; opts = Float64[]; execs = Float64[]
    for _ in 1:5
        t0 = time(); r = galley(q; verbose=0); push!(walls, time() - t0)
        push!(opts, r.opt_time); push!(execs, r.execute_time)
    end
    sort!(walls); sort!(opts); sort!(execs)
    res["warm_wall_median"] = walls[3]; res["warm_opt_median"] = opts[3]; res["warm_exec_median"] = execs[3]
    val = r.value[1]
    arr = val isa Number ? [Float64(val)] : Array{Float64}(Array(val))
    npzwrite(joinpath(DIR, case * "_galley_out.npy"), arr .* spec["scale"])
    res["out_shape"] = collect(size(arr))
    try
        res["logical_plan"] = sprint(io -> show(io, galley(q; output_logical_plan=true)))[1:min(end, 4000)]
    catch e
        res["logical_plan"] = "unavailable: " * sprint(showerror, e)[1:min(end, 300)]
    end
    open(joinpath(DIR, case * "_galley.json"), "w") do io
        JSON.print(io, res, 1)
    end
    println(case, " cold ", round(res["cold_wall"], digits=3), "s warm ", round(res["warm_wall_median"], digits=4),
            "s (opt ", round(res["warm_opt_median"], digits=4), " exec ", round(res["warm_exec_median"], digits=4), ")")
end

for case in ARGS
    try
        run_case(case)
    catch e
        msg = sprint(showerror, e)
        println(case, " FAILED: ", msg[1:min(end, 600)])
        open(joinpath(DIR, case * "_galley.json"), "w") do io
            JSON.print(io, Dict("case" => case, "error" => msg[1:min(end, 4000)]), 1)
        end
    end
end
