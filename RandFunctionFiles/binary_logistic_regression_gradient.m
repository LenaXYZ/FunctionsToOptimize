function grad = binary_logistic_regression_gradient(W, X, Y)
%
% BINARY_LOGISTIC_REGRESSION_GRADIENT
%   Computes binary logistic regression gradient.
%
%   Parameters
%     W: target weight vector   (dims X 1)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%

[dims, Wcols] = size(W);
assert(Wcols == 1);
[examples, dims] = size(X);
margins = X*W;  % (examples X 1)

grad = (-Y' ./ (1 + exp(Y'.*margins)))' * X / (examples );