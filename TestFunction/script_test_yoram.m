%% Create function
init_params_Yoram;
data = generate_data(data_params);
data_for_train.X= data.Xtrain;
data_for_train.Y= data.Ytrain;
problem.trainFunction = YoramFunction(data_for_train, model_params, 'yoram_train');
thefunc = problem.trainFunction;

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
