function loss = linear_regression_sq_loss(W, X, Y)
%
% LINEAR_REGRESSION_SQ_LOSS
%   Computes linear regression squared loss.
%
%   Parameters:
%     W: target weight vector   (dims X 1)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%

[dims, Wcols] = size(W);
assert(Wcols == 1);
[examples, dims] = size(X);
loss = sum((X*W - Y').^2) / (2 * examples);
