function loss = binary_logistic_regression_loss(W, X, Y)
%
% BINARY_LOGISTIC_REGRESSION_LOSS
%   Computes binary logistic regression loss.
%
%   Parameters:
%     W: target weight vector   (dims X 1)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%

[dims, Wcols ] = size(W);
assert(Wcols == 1);
[examples, dims] = size(X);
loss = sum( log(1 + exp(-Y' .* (X*W))) ) / examples;
