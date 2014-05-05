function hessian = binary_logistic_regression_hessian(W, X, Y)
%
% Inverse hessian
%
%   Parameters
%     W: target weight vector   (1 X dims)
%     X: data                   (examples X dims)
%     Y: labels                 (1 X examples)
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