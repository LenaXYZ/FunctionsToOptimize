function [trainFunction,testFunction] = create_RCV1()
fprintf('Creating RCV1 train and test functions...\n');

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
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
trainFunction = RCV1like(Data_Train, 'RCV1_train');
testFunction = RCV1like(Data_Test, 'RCV1_test');

fprintf('RCV1 train and test functions created\n');
end