using Test
using Random
using NCDatasets
using OrderedCollections
using NCRegressionTests
Random.seed!(123)

@testset "NCRegressionTests - perfect match" begin

    sz = (123, 145)
    data = randn(MersenneTwister(152), sz)

    filenameA = tempname()
    filenameB = tempname()

    NCDataset(filenameA, "c") do ds
        defDim(ds, "lon", sz[1])
        defDim(ds, "lat", sz[2])
        v = defVar(ds, "var", Float64, ("lon", "lat"))
        v[:, :] = data
        v = defVar(ds, "var2", Float64, ("lon", "lat"))
        v[:, :] = 2 * data
    end
    NCDataset(filenameB, "c") do ds
        defDim(ds, "lon", sz[1])
        defDim(ds, "lat", sz[2])
        v = defVar(ds, "var", Float64, ("lon", "lat"))
        v[:, :] = data
        v = defVar(ds, "var2", Float64, ("lon", "lat"))
        v[:, :] = 2 * data
    end

    best_mse = OrderedCollections.OrderedDict("var" => 0, "var2" => 0)

    computed_mse = NCRegressionTests.compute_mse(;
        job_name = "NCTest",
        best_mse,
        ds_filename_computed = filenameA,
        ds_filename_reference = filenameB,
    )
    @test computed_mse["var"] == 0
    @test computed_mse["var2"] == 0

    NCRegressionTests.test_mse(computed_mse, best_mse)

end

@testset "NCRegressionTests - Mismatch" begin

    sz = (123, 145)
    data = randn(MersenneTwister(152), sz)

    filenameA = tempname()
    filenameB = tempname()

    NCDataset(filenameA, "c") do ds
        defDim(ds, "lon", sz[1])
        defDim(ds, "lat", sz[2])
        v = defVar(ds, "var", Float64, ("lon", "lat"))
        v[:, :] = data .+ 1
        v = defVar(ds, "var2", Float64, ("lon", "lat"))
        v[:, :] = 2 * data .+ 2
    end
    NCDataset(filenameB, "c") do ds
        defDim(ds, "lon", sz[1])
        defDim(ds, "lat", sz[2])
        v = defVar(ds, "var", Float64, ("lon", "lat"))
        v[:, :] = data
        v = defVar(ds, "var2", Float64, ("lon", "lat"))
        v[:, :] = 2 * data
    end

    best_mse = OrderedCollections.OrderedDict("var" => 2.8633275236008583e+04, "var2" => 0)

    computed_mse = NCRegressionTests.compute_mse(;
        job_name = "NCTest",
        best_mse,
        ds_filename_computed = filenameA,
        ds_filename_reference = filenameB,
    )

    @test computed_mse["var"] ≈ 2.8633275236008583e+04
    @test computed_mse["var2"] ≈ 2.8633275236008583e+04

    best_mse = OrderedCollections.OrderedDict("var" => 2.8633275236008583e+04, "var2" => 0, "no_key" => 1)

    err = ErrorException("No key no_key for mse computation.")
    @test_throws err NCRegressionTests.compute_mse(;
        job_name = "NCTest",
        best_mse,
        ds_filename_computed = filenameA,
        ds_filename_reference = filenameB,
    )

end
