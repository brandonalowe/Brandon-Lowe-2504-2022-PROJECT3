using DataFrames, CSV

function import_data(file)
    csv_file = CSV.File(file; missingstring="NA")
    df = DataFrame(csv_file)
    return df
end


# here for testing the function
df = import_data("data/Melbourne_housing_FULL.csv")
first(df, 5)