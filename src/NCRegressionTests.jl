module NCRegressionTests

using Test
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

"""
    compute_mse(;
        job_name::String,
        reference_mse::OrderedCollections.OrderedDict,
        ds_filename_computed::String,
        ds_filename_reference::String,
        varname::Function = s -> s,
        debug_print = false,
        compare_mse = default_compare_mse,
    )

Returns a `Dict` of mean-squared errors between
`NCDataset`s `ds_filename_computed` and
`ds_filename_reference` for all keys in `reference_mse`.
Keys in `reference_mse` may directly map to keys in
the `NCDataset`s, or they may be mapped to the keys
via `varname`.
"""
function compute_mse(;
    job_name::String,
    reference_mse::OrderedCollections.OrderedDict,
    ds_filename_computed::String,
    ds_filename_reference::String,
    varname::Function = s -> s,
    debug_print = false,
    compare_mse = default_compare_mse,
)
    @assert isfile(ds_filename_computed)
    @assert isfile(ds_filename_reference)

    computed_mse = NC.Dataset(ds_filename_computed, "r") do ds_computed
        NC.Dataset(ds_filename_reference, "r") do ds_reference
            _compute_mse(job_name, reference_mse, ds_computed, ds_reference, varname, debug_print, compare_mse)
        end
    end
end

function _compute_mse(job_name, reference_mse, ds_computed, ds_reference, varname, debug_print, compare_mse)

    mse = OrderedCollections.OrderedDict()
    # Ensure z_tcc and fields are consistent lengths:
    best_keys = keys(reference_mse)
    n_keys = length(best_keys)
    variables = Vector{String}(undef, n_keys)
    variables .= varname.(best_keys)
    computed_mse = zeros(n_keys)
    table_reference_mse = zeros(n_keys)
    mse_reductions = zeros(n_keys)
    data_scales_now = zeros(n_keys)
    data_scales_ref = zeros(n_keys)

    for (i, key) in enumerate(best_keys)
        str_key = varname(key)
        if debug_print
            @info "Computing mse for `$key`, `$str_key`"
        end
        data_computed_arr = vec(get_nc_data(ds_computed, str_key))
        data_reference_arr = vec(get_nc_data(ds_reference, str_key))

        # Interpolate data
        # Compute data scale
        data_scale_now = sum(abs.(data_computed_arr)) / length(data_computed_arr)
        data_scale_ref = sum(abs.(data_reference_arr)) / length(data_reference_arr)
        data_scales_now[i] = data_scale_now
        data_scales_ref[i] = data_scale_ref

        # Compute mean squared error (mse)
        mse_single_var = sum((data_computed_arr .- data_reference_arr) .^ 2)
        scaled_mse = mse_single_var / data_scale_ref^2 # Normalize by data scale

        mse[key] = scaled_mse
        mse_reductions[i] = (reference_mse[key] - mse[key]) / reference_mse[key] * 100
        table_reference_mse[i] = reference_mse[key]
        computed_mse[i] = mse[key]
    end

    # Tabulate output
    header = (
        ["Variable", "Data scale", "Data scale", "MSE", "MSE", "MSE"],
        ["", "Computed", "Reference", "Computed", "Reference", "Reduction (%)"],
    )

    table_data = hcat(variables, data_scales_now, data_scales_ref, computed_mse, table_reference_mse, mse_reductions)

    hl_worsened_mse = PrettyTables.Highlighter(
        (data, i, j) -> !compare_mse(data[i, 4], data[i, 5]) && j == 4,
        PrettyTables.crayon"red bold",
    )
    hl_worsened_mse_reduction = PrettyTables.Highlighter(
        (data, i, j) -> !compare_mse(data[i, 4], data[i, 5]) && j == 6,
        PrettyTables.crayon"red bold",
    )
    hl_improved_mse = PrettyTables.Highlighter(
        (data, i, j) -> compare_mse(data[i, 4], data[i, 5]) && j == 6,
        PrettyTables.crayon"green bold",
    )
    @info "Regression tables for `$job_name`"
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

default_compare_mse(computed_mse::Real, reference_mse::Real) =
    reference_mse - sqrt(eps()) ≤ computed_mse ≤ reference_mse + sqrt(eps())

# Sometimes we don't have variables to compare against, so
# we have default behavior for handling these cases.
# Users may want to overload these methods for their own use-case.
default_compare_mse(computed_mse::String, reference_mse::String) = true
default_compare_mse(computed_mse::Real, reference_mse::String) = true
default_compare_mse(computed_mse::String, reference_mse::Real) = true

function test_mse(computed_mse, reference_mse, compare_mse = default_compare_mse)
    @testset "Regression tests" begin
        for key in keys(reference_mse)
            mse_not_regressed = compare_mse(computed_mse[key], reference_mse[key])
            if !mse_not_regressed
                @info "Regression failed for `$key`. (computed | best) mse: `$(computed_mse[key])` | `$(reference_mse[key])`"
            end
            @test mse_not_regressed
        end
    end
end

end # module
