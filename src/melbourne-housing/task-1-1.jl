# Script for task 1.1

using Pkg, Plots, StatsPlots, StatsBase
Pkg.activate(".")

include("../../src/import-csv.jl")

housing_data_df = import_data("data/Melbourne_housing_FULL.csv")

# println(names(housing_data_df))

histogram(housing_data_df.Rooms, legend=false, xlabel = "Number of rooms")


