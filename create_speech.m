function [trainFunction,testFunction] = create_speech()
fprintf('Creating Speech train and test functions...\n');

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
%%%%%%%% Configurable parameters

fractionOfDataUsedForTraining = .75;
seedForPermutingData = 1;

%%%%%%%% End configurable parameters

% Load speech data
data=load('Speech_Data.mat');
data.M(:,1) = data.M(:,1)+1;

% Separate into training and testing data
numDataPts  = size(data.M,1);
numTrainPts = ceil(numDataPts*fractionOfDataUsedForTraining);
aRandomStreamForPermutingDataIndices = RandStream('mt19937ar','Seed',seedForPermutingData);
permutedIndices = randperm(aRandomStreamForPermutingDataIndices,numDataPts);
Data_Train  = data.M(permutedIndices(1:numTrainPts),:);
Data_Test   = data.M(permutedIndices((numTrainPts+1):end),:);

% Create train and test function
trainFunction = SumNegLogLikelihood(Data_Train, 'speech_train');
testFunction = SumNegLogLikelihood(Data_Test, 'speech_test');

fprintf('Speech train and test functions created\n');
end