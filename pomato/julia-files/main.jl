# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018 
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v0.6.2 (required)
# ----------------------------------
# 
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -------------------------------------------------------------------------------------------------

# To use file both in POMATO and as script
if length(ARGS) > 0
    WDIR = ARGS[1]
else
    println("No arguments passed, running as script in pwd()")
    WDIR = pwd()
end


# Invoke Required packages (todo: make this a macro)
try
    using DataFrames
catch
    Pkg.add("DataFrames")
    using DataFrames
end

try
    using CSV
catch
    Pkg.add("CSV")
    using CSV
end

try 
    using JSON
catch
    Pkg.add("JSON")
    using JSON
end

try 
    using DataStructures
catch
    Pkg.add("DataStructures")
    using DataStructures
end

try 
    using JuMP
catch
    Pkg.add("JuMP")
    using JuMP
end

try 
    using Clp
catch
    Pkg.add("Clp")
    using Clp
end


include("src/tools.jl")
include("src/typedefinitions.jl")
include("src/read_data.jl")
include("src/model.jl")
include("src/setdefinitions.jl")

plants, plants_in_ha, plants_at_node, plants_in_zone, availabilites, 
dc_lines, nodes, slack_zones, zones, heatareas, cbco, ntc, 
model_horizon, opt_setup = read_model_data(WDIR*"/data/")


#Run Dispatch Model
out = build_and_run_model(plants, plants_in_ha, plants_at_node, plants_in_zone, availabilites, 
                   nodes, zones, heatareas, ntc, dc_lines, slack_zones, cbco,
                   model_horizon, opt_setup)


println("DONE")
