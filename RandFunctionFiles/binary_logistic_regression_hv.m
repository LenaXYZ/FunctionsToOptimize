function hv = binary_logistic_regression_hv(W, X, Y, v)
%
% BINARY_LOGISTIC_REGRESSION_GRADIENT
%   Computes binary logistic regression gradient.
%
%   Parameters
%     W: target weight vector   (dims X 1)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
%     v: vector to multiply by
%

[~,Wcols] = size(W);
assert(Wcols == 1);
[examples, dims] = size(X);
hessian=zeros(dims,dims);
expinformula = exp(-Y'.*(X*W)); % 1 X examples
everythingButX = (Y').^2.*(expinformula)./(expinformula+1).^2;
for i = 1:examples
    hessian = hessian+X(i,:)'*X(i,:)*everythingButX(i);
end
hessian=hessian/examples;

%fprintf('Cond number: %e\n',cond(hessian));
hv =   hessian*v;
