function B = linear_regression_huber_curvature(X)
%
% LINEAR_REGRESSION_HUBER_CURVATURE
%   Computes curvature over the dimensions for examples in X for
%   linear regression with Huber loss.  Using the squared loss
%   conservative estimate as it upper bounds.
%
%   Returns:
%     B: betas                  (1 X dims)
%
%   Parameters:
%     X: data                   (examples X dims)
%

[examples, dims] = size(X);

B = diag(X' * X) / examples;
