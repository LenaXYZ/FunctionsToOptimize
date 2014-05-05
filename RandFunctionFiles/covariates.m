function A = covariates(rows, cols, mu)
%
% COVARIATES
%   Returns:
%     A ~ N(mu, sigma) + noise where dim(A)=(rows, cols).
%

noise = rand(1)/10;
A = repmat(ones(1, cols).*mu, rows, 1);

% Create a sparse PSD covariance matrix.
S = sparsify(randn(cols), 0.9);
S(1:cols+1:cols*cols) = randn(1, cols);
S = S' * S;

A = A + ...
    randn(rows, cols) * chol(S) + ...   % covar
    randn(rows, cols) * rand(1)/10;     % noise
