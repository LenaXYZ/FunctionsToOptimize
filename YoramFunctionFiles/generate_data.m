function data = generate_data(params,percentTrain)
%
% GENERATE_DATA
%   Generate parameters for learning problems.
%
%   Returns:
%     data.optimalSolution: target weight vector   (classes  X dims)
%     data.X: data                   (examples X dims)
%     data.Y: labels                 (classes  X examples)
%   For the case of binary classification, classes=1.
%
%   Configurable parameters include:
%   - params.classes:      num classes.
%   - params.dims:         num dimensions.
%   - params.x.mu:         mean around example vectors.
%   - params.x.size:       num examples.
%   - params.x.sparsity:   sparsity of generated X matrix.
%   - params.w.dist:       1=U[-1,1]; 2=N(mu,sigma)+noise.
%   - params.w.mu:         mean around weight vectors.
%   - params.w.sparsity:   sparsity of generated W matrix.
%   - params.y.regression: 1=linear;         2=logistic
%   - params.y.type:       1=deterministic;  2=stochastic

assert(params.dims > 0);
assert(params.x.size > 0);
assert(params.w.sparsity >= 0 && params.w.sparsity < 1);
assert(params.w.dist == 1 || params.w.dist == 2);
assert(params.y.regression == 1 || params.y.regression == 2);
assert(params.y.type == 1 || params.y.type == 2);
% For logistic regression, we currently only support binary classes.
assert((params.y.regression == 1 && params.classes == 1) || ...
    (params.y.regression == 2 && params.classes == 2));

Wrows = params.classes;
if Wrows == 2
    Wrows = 1;
end

if params.w.dist == 1  % Uniform[-1,1]
    optimalSolution = rand(Wrows, params.dims)*2 - 1;
else                   % Normal[mu,sigma]
    optimalSolution = covariates(Wrows, params.dims, params.w.mu);
end;
optimalSolution = sparsify(optimalSolution, params.w.sparsity);
X = sparsify(covariates(params.x.size, params.dims, params.x.mu), ...
    params.x.sparsity);

Y = optimalSolution * X' + randn(Wrows, params.x.size) * rand(1)/10;
if params.y.regression == 2  % Logistic
    if Wrows > 2
        norms = repmat(1./sum(exp(Y)), Wrows, 1);
        Y = exp(Y) .* norms;
        % Multi-class labels.
        if params.y.type == 1  % Deterministic
            [yM, yI] = max(Y);
            Y = yI;
        else                   % Stochastic
            balls = rand(1, params.x.size);
            cY = cumsum(Y);
            Y = [];
            for j=1:params.x.size
                Y = [Y find(cY(:,j) >= balls(j), 1)];
            end
        end
        
        % Make Y into a classes X examples matrix where 1=relevant, 0=irrelevant.
        col_offsets = [1:Wrows:Wrows*params.x.size]-1;
        yY = -1*ones(Wrows, params.x.size);
        yY(Y + col_offsets) = 1;
        Y = yY;
    else
        % Binary labels.
        Y = sign(Y);
    end
end

train_max_idx = percentTrain*params.x.size;

data.optimalSolution = optimalSolution;
data.Xtrain = X(1:train_max_idx,:);
data.Ytrain = Y(:, 1:train_max_idx);
data.Xtest  = X(train_max_idx+1:params.x.size,:);
data.Ytest  = Y(:, train_max_idx+1:params.x.size);
