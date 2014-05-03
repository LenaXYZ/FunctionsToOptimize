
fractionOfDataUsedForTraining = .75;
seedForPermutingData = 1;

%%%%%%%% End configurable parameters

% Load speech data
data=load('~/DropboxATLAB\CovarianceHessian\Data\Speech_Data.mat');
data.M(:,1) = data.M(:,1)+1;

% Separate into training and testing data
numDataPts  = size(data.M,1);
numTrainPts = ceil(numDataPts*fractionOfDataUsedForTraining);
numTestPts  = numDataPts - numTrainPts;
aRandomStreamForPermutingDataIndices = RandStream('mt19937ar','Seed',seedForPermutingData);
permutedIndices = randperm(aRandomStreamForPermutingDataIndices,numDataPts);
Data_Train  = data.M(permutedIndices(1:numTrainPts),:);
Data_Test   = data.M(permutedIndices((numTrainPts+1):end),:);

% Create train and test function
thefunc = SumNegLogLikelihood(Data_Train, 'speech_train');

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
