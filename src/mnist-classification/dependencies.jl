using MLDatasets, LinearAlgebra, Random, Distributions, Flux, BenchmarkTools, Plots, Measures, StatsBase, DecisionTree, Statistics

using Flux: params, onehotbatch, crossentropy, update!, binarycrossentropy

include("linear-regression.jl")
include("logistic-regression.jl")
include("logistic-softmax-regression.jl")


MNIST_train_imgs, MNIST_train_labels = MNIST.traindata(Float64);    # Training data for MNIST ML dataset
MNIST_test_imgs, MNIST_test_labels = MNIST.testdata(Float64);       # Testing data for MNIST ML dataset

fMNIST_train_imgs, fMNIST_train_labels = FashionMNIST.traindata(Float64);   # Training data for Fashion MNIST ML dataset
fMNIST_test_imgs, fMNIST_test_labels = FashionMNIST.testdata(Float64);      # Testing data for Fashion MNIST ML dataset