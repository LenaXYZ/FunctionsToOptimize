function [trainFunction,testFunction] = create_tfdfRCV1(classIndex)
fprintf('Creating tfdfRCV1 train and test functions...\n');
load('tfidfRCV1');

s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
p = .75;
seedForPermutingData = 1;
numDataPts  = size(data,1);
numTrainPts = ceil(numDataPts*p);
seed        = RandStream('mt19937ar','Seed',seedForPermutingData);
rp          = randperm(seed,numDataPts);
Data_Train  = data(rp(1:numTrainPts),:);
Data_Test   = data(rp((numTrainPts+1):end),:);
% Create train and test function
trainFunction = tfdfRCV1(Data_Train, 'tfdfRCV1_train',classIndex);
testFunction = tfdfRCV1(Data_Test, 'tfdfRCV1_test',classIndex);

fprintf('tfdfRCV1 train and test functions created\n');
end