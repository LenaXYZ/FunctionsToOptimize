%% Create function
optsYoram.fractionToTrain = 7000/10000;
optsYoram.regression =2;
[thefunc, testFunctionU]=create_randFunction(optsYoram);
thefunc = L2regFunction(thefunc, 1e-2);

%% Test f and g
test_f_g(thefunc,zeros(50,1),1)
test_f_g(thefunc,ones(50,1),1)
test_f_g(thefunc,-ones(50,1),1)

%% Test sampling
test_f_g(thefunc,zeros(50,1),2)
test_f_g(thefunc,ones(50,1),2)
test_f_g(thefunc,-ones(50,1),2)

%% Test deterministic sampling
test_f_g(thefunc,zeros(50,1),3)
test_f_g(thefunc,ones(50,1),3)
test_f_g(thefunc,-ones(50,1),3)

%% Test hv
test_f_g(thefunc,zeros(50,1),4)
test_f_g(thefunc,ones(50,1),4)
test_f_g(thefunc,-ones(50,1),4)
test_f_g(thefunc,randn(50,1),4)