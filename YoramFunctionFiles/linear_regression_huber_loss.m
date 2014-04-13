function loss = linear_regression_huber_loss(W, X, Y, delta)
%
% LINEAR_REGRESSION_HUBER_LOSS
%   Computes linear regression Huber loss.
%
%   Parameters:
%     W: target weight vector   (1 X dims)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%
%   Optional Parameters:
%     delta: Huber loss parameter controls extent of quadratic/linear component.
%

if nargin == 3
  delta = 1;
end

[Wrows, dims] = size(W);
assert(Wrows == 1);
[examples, dims] = size(X);
residual = W*X' - Y;

% Huber loss quadratic component.
case_1 = find(abs(residual) <= delta);
residual_1 = sum(residual(case_1).^2) / 2;

% Huber loss linear component.
case_2 = find(abs(residual) >  delta);
residual_2 = sum(abs(residual(case_2)) - delta/2) * delta;

loss = (residual_1 + residual_2) / examples;
