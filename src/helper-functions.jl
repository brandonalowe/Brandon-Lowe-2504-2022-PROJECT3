function import_data(file)
    csv_file = CSV.File(file; missingstring=["", "#N/A"])
    df = DataFrame(csv_file)
    return df
end;
