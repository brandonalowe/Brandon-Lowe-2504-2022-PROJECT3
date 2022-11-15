using DataFrames, CSV, Plots, StatsPlots, StatsBase, HTTP, SplitApplyCombine, GLM, Random

include("helper-functions.jl") # some helper functions I wrote to help keep notebook clean

housing_data_df = import_data("data/Melbourne_housing_FULL.csv")