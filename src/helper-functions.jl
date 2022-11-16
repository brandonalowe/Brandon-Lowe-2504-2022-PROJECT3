function import_data(file)
    csv_file = CSV.File(file; missingstring=["", "#N/A"], dateformat="dd/mm/yyyy")
    df = DataFrame(csv_file)
    return df
end;
