function grad = linear_regression_huber_gradient(W, X, Y, T, delta)
%
% LINEAR_REGRESSION_HUBER_GRADIENT
%   Computes linear regression gradient for Huber loss.
%
%   Parameters:
%     W: target weight vector   (1 X dims)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%     T: coordinates to step    (1 X num_coordinates)
%
%   Optional Parameters:
%     delta: Huber loss parameter controls extent of quadratic/linear component.
%

if nargin == 4
  delta = 1;
end

[Wrows, dims] = size(W);
assert(Wrows == 1);
[examples, dims] = size(X);
residual = W*X' - Y;  % (1 X examples)

% Huber loss quadratic component.
case_1 = find(abs(residual) <= delta);
grad_1 = residual(case_1) * X(case_1,T);

% Huber loss linear component.
case_2 = find(abs(residual) >  delta);
grad_2 = delta * sign(residual(case_2)) * X(case_2,T);

grad = (grad_1 + grad_2) / (examples * length(T));
