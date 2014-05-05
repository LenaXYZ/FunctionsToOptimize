function grad = linear_regression_sq_gradient(W, X, Y)
%
% LINEAR_REGRESSION_SQ_GRADIENT
%   Computes linear regression gradient for squared loss.
%
%   Parameters:
%     W: target weight vector   (dims X 1)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%

[dims, Wcols] = size(W);
assert(Wcols == 1);
[examples, dims] = size(X);
residual = X*W - Y';  % (examples x 1)
grad = (X'*residual)'   / (examples );
