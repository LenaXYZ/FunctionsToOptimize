%% Load RCV1
data        = load('dataRCV1.mat');
p = .75;
seedForPermutingData = 1;
numDataPts  = size(data.RCV1hold,1); 
numTrainPts = ceil(numDataPts*p); 
seed        = RandStream('mt19937ar','Seed',seedForPermutingData);
rp          = randperm(seed,numDataPts);
Data_Train  = data.RCV1hold(rp(1:numTrainPts),:);
Data_Test   = data.RCV1hold(rp((numTrainPts+1):end),:);

% Create train and test function
thefunc = RCV1like(Data_Train, 'RCV1_train');

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
