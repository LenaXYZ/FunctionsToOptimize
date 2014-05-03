function hv = linear_regression_sq_hv(W, X, Y, v)
%
% BINARY_LOGISTIC_REGRESSION_GRADIENT
%   Computes binary logistic regression gradient.
%
%   Parameters
%     W: target weight vector   (1 X dims)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%     v: vector to multiply by
%

[examples, dims] = size(X);

hv= (X' * X/ examples)*v;