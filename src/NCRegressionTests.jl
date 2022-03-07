module NCRegressionTests

import OrderedCollections
import NCDatasets
const NC = NCDatasets
import PrettyTables

function get_nc_data(ds, var::String)
    if haskey(ds, var)
        return ds[var]
    else
        for key in keys(ds.group)
            if haskey(ds.group[key], var)
                return ds.group[key][var]
            end
        end
    end
    error("No key $var for mse computation.")
    return nothing
end

function compute_mse(; job_name::String, best_mse, ds_filename_computed::String, ds_filename_reference::String)
    @assert isfile(ds_filename_computed)
    @assert isfile(ds_filename_reference)

    computed_mse = NC.Dataset(ds_filename_computed, "r") do ds_computed
        NC.Dataset(ds_filename_reference, "r") do ds_reference
            _compute_mse(job_name, best_mse, ds_computed, ds_reference)
        end
    end
end

function _compute_mse(job_name, best_mse, ds_computed, ds_reference)

    mse = OrderedCollections.OrderedDict()
    # Ensure z_tcc and fields are consistent lengths:
    best_keys = keys(best_mse)
    n_keys = length(best_keys)
    variables = Vector{String}(undef, n_keys)
    variables .= best_keys
    computed_mse = zeros(n_keys)
    table_best_mse = zeros(n_keys)
    mse_reductions = zeros(n_keys)
    data_scales_now = zeros(n_keys)
    data_scales_ref = zeros(n_keys)

    for (i, tc_var) in enumerate(variables)
        data_computed_arr = Array(get_nc_data(ds_computed, tc_var))'
        data_reference_arr = Array(get_nc_data(ds_reference, tc_var))'

        # Interpolate data
        # Compute data scale
        data_scale_now = sum(abs.(data_computed_arr)) / length(data_computed_arr)
        data_scale_ref = sum(abs.(data_reference_arr)) / length(data_reference_arr)
        data_scales_now[i] = data_scale_now
        data_scales_ref[i] = data_scale_ref

        # Compute mean squared error (mse)
        mse_single_var = sum((data_computed_arr .- data_reference_arr) .^ 2)
        scaled_mse = mse_single_var / data_scale_ref^2 # Normalize by data scale

        mse[tc_var] = scaled_mse
        mse_reductions[i] = (best_mse[tc_var] - mse[tc_var]) / best_mse[tc_var] * 100
        table_best_mse[i] = best_mse[tc_var]
        computed_mse[i] = mse[tc_var]
    end

    # Tabulate output
    header = (
        ["Variable", "Data scale", "Data scale", "MSE", "MSE", "MSE"],
        ["", "Computed", "Reference", "Computed", "Best", "Reduction (%)"],
    )

    table_data = hcat(variables, data_scales_now, data_scales_ref, computed_mse, table_best_mse, mse_reductions)

    hl_worsened_mse = PrettyTables.Highlighter(
        (data, i, j) -> !sufficient_mse(data[i, 4], data[i, 5]) && j == 4,
        PrettyTables.crayon"red bold",
    )
    hl_worsened_mse_reduction = PrettyTables.Highlighter(
        (data, i, j) -> !sufficient_mse(data[i, 4], data[i, 5]) && j == 6,
        PrettyTables.crayon"red bold",
    )
    hl_improved_mse = PrettyTables.Highlighter(
        (data, i, j) -> sufficient_mse(data[i, 4], data[i, 5]) && j == 6,
        PrettyTables.crayon"green bold",
    )
    @info "Regression tables for $job_name"
    PrettyTables.pretty_table(
        table_data;
        header,
        formatters = PrettyTables.ft_printf("%.16e", 4:5),
        header_crayon = PrettyTables.crayon"yellow bold",
        subheader_crayon = PrettyTables.crayon"green bold",
        highlighters = (hl_worsened_mse, hl_improved_mse, hl_worsened_mse_reduction),
        crop = :none,
    )

    return mse
end

sufficient_mse(computed_mse::T, best_mse::T) where {T <: Real} = computed_mse <= best_mse + sqrt(eps())

end # module
