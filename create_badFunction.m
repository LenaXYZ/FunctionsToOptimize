function [trainFunction,testFunction] = create_badFunction(opts)
fprintf('Creating bad train and test functions...\n');
s = RandStream('mt19937ar','Seed',3);
RandStream.setGlobalStream(s);% See 'generate_data(params)' on how the following params are used.

meanA = randn(1);
stdA = rand(1);

A = stdA.*randn(opts.m,2*opts.n/3) + meanA;

B = randn(opts.m,1);

TAU = rand(opts.m,1);

GAMMA = rand(opts.m,1);


kk = floor(0.75*opts.m);
trainFunction = BadFunction(A(1:kk,:),B(1:kk,:),TAU(1:kk,:),GAMMA(1:kk,:), 'bad_train');
testFunction = BadFunction(A(kk+1:end,:),B(kk+1:end,:),TAU(kk+1:end,:),GAMMA(kk+1:end,:), 'bad_test');
fprintf('Bad train and test functions created\n');
end