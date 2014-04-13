function B = binary_logistic_regression_curvature(X)
%
% BINARY_LOGISTIC_REGRESSION_CURVATURE
%   Computes curvature over the dimensions for examples in X for
%   binary logistic regression.
%
%   Returns:
%     B: betas                  (1 X dims)
%
%   Parameters:
%     X: data                   (examples X dims)
%

[examples, dims] = size(X);
B = diag(X' * X) / (4 * examples);
