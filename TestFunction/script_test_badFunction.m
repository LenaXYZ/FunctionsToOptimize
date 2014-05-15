%% Create function
clc;clear;close all;
optsBad.m=4; %div by 4
optsBad.n=6;  %div by 3
[trainFunction,testFunction] = create_badFunction(optsBad);
thefunc = trainFunction;

%% Test f and g
test_f_g(thefunc,zeros(thefunc.numVariables,1),1)
test_f_g(thefunc,ones(thefunc.numVariables,1),1)
test_f_g(thefunc,-ones(thefunc.numVariables,1),1)

%% Test sampling
test_f_g(thefunc,zeros(thefunc.numVariables,1),2)
test_f_g(thefunc,ones(thefunc.numVariables,1),2)
test_f_g(thefunc,-ones(thefunc.numVariables,1),2)

%% Test deterministic sampling
test_f_g(thefunc,zeros(thefunc.numVariables,1),3)
test_f_g(thefunc,ones(thefunc.numVariables,1),3)
test_f_g(thefunc,-ones(thefunc.numVariables,1),3)

%% Test hv
test_f_g(thefunc,zeros(thefunc.numVariables,1),4)
test_f_g(thefunc,ones(thefunc.numVariables,1),4)
test_f_g(thefunc,-ones(thefunc.numVariables,1),4)
test_f_g(thefunc,randn(thefunc.numVariables,1),4)