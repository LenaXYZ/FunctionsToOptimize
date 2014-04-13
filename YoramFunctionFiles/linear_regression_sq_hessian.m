function hessian = linear_regression_sq_hessian(W, X, Y)
%
%Inverse hessian
%
%   Parameters
%     W: target weight vector   (1 X dims)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%     v: vector to multiply by
%

[examples, dims] = size(X);
hessian = X' * X/ examples;