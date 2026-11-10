# Galley runner for bench/systems.py (main evaluation). Usage:
#   julia --project=D:\runs\knitplan\galley_env galley_runner.jl <dir> <reps>
# <dir>/spec.json + <dir>/factors.npz hold the pushed-down factors as COO (one 0-based coordinate array per index,
# T<p>_c<d>, and values T<p>_v). Each factor is converted to a Finch tensor with SparseList levels (as in D2/E1-G);
# the query is Mat(outs..., Agg(+, 0.0, cut indices..., MapJoin(*, factors...))).
# One cold run (includes Julia compilation), then <reps> warm runs. Times per warm run: t_conv (COO -> Finch tensors)
# and t_galley (galley() wall: its optimizer + execution, also reported separately). Medians go to <dir>/result.json,
# the answer (stored entries of the last warm run) to <dir>/answer.npz.
using Galley, Galley.Optimizer, NPZ, JSON, Statistics   # Galley vendors Finch (Tensor, SparseList, fsparse, ffindnz)

const dir = ARGS[1]
const reps = parse(Int, ARGS[2])
const spec = JSON.parsefile(joinpath(dir, "spec.json"))
const data = npzread(joinpath(dir, "factors.npz"))

function build_inputs()
    inputs = []
    for t in spec["tensors"]
        k = t["key"]; dims = Tuple(Int.(t["dims"]))
        I = ntuple(d -> Vector{Int}(data["$(k)_c$(d - 1)"]) .+ 1, length(dims))
        V = Vector{Float64}(data["$(k)_v"])
        coo = fsparse(I..., V, dims)
        lvl = Element(0.0)
        for _ in 1:length(dims)
            lvl = SparseList(lvl)
        end
        push!(inputs, Input(Tensor(lvl, coo), Symbol.(t["idx"])...))
    end
    return inputs
end

function make_query(inputs)
    outs = Symbol.(spec["outputs"])
    allidx = unique(vcat([Symbol.(t["idx"]) for t in spec["tensors"]]...))
    summed = [i for i in allidx if !(i in outs)]
    expr = MapJoin(*, inputs...)
    body = isempty(summed) ? expr : Agg(+, 0.0, summed..., expr)
    return isempty(outs) ? Mat(body) : Mat(outs..., body)
end

res = Dict{String,Any}()
t0 = time(); r = galley(make_query(build_inputs()); verbose=0); res["cold_total"] = time() - t0
convs = Float64[]; walls = Float64[]; opts = Float64[]; execs = Float64[]
for _ in 1:reps
    t1 = time(); inputs = build_inputs(); push!(convs, time() - t1)
    q = make_query(inputs)
    t2 = time(); global r = galley(q; verbose=0); push!(walls, time() - t2)
    push!(opts, r.opt_time); push!(execs, r.execute_time)
end
res["t_conv"] = median(convs); res["t_galley"] = median(walls)
res["t_galley_opt"] = median(opts); res["t_galley_exec"] = median(execs)
res["t_conv_all"] = convs; res["t_galley_all"] = walls
val = r.value[1]
out = Dict{String,Any}()
if val isa Number
    out["v"] = [Float64(val)]
    res["scalar"] = true
else
    nz = ffindnz(val)
    for d in 1:(length(nz) - 1)
        out["o$(d - 1)"] = Vector{Int64}(nz[d]) .- 1
    end
    out["v"] = Vector{Float64}(nz[end])
    res["scalar"] = false
end
npzwrite(joinpath(dir, "answer.npz"), out)
open(joinpath(dir, "result.json"), "w") do io
    JSON.print(io, res, 1)
end
