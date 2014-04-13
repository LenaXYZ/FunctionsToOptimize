function [trainFunction,testFunction] = create_yoram(percentTrain)
fprintf('Creating Yoram train and test functions...\n');
s = RandStream('mt19937ar','Seed',3);
RandStream.setGlobalStream(s);
init_params_Yoram;
data = generate_data(data_params,percentTrain);
data_for_train.X= data.Xtrain;
data_for_train.Y= data.Ytrain;
trainFunction = YoramFunction(data_for_train, model_params, 'yoram_train');
data_for_test.X= data.Xtest;
data_for_test.Y= data.Ytest;
testFunction = YoramFunction(data_for_test, model_params, 'yoram_test');

save('dataHW1','data')
fprintf('Yoram train and test functions created\n');
end