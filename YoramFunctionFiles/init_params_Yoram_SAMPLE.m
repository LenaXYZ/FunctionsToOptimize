% See 'generate_data(params)' on how the following params are used.
data_params.dims         = 50;
data_params.x.mu         = 0;
data_params.x.size       = 10000;
data_params.x.sparsity   = 0.25;
data_params.w.dist       = 2;
data_params.w.mu         = 0;
data_params.w.sparsity   = 0.25;
data_params.y.regression = 2;
data_params.y.type       = 1;
if data_params.y.regression == 1      % linear regression
  data_params.classes = 1;
elseif data_params.y.regression == 2  % logistic regression
  data_params.classes = 2;
end

% See 'coordinate_descent(params)' on how the following params are used.
model_params.min_epochs       = 100;
model_params.term_criterion   = 1e-6;
model_params.epoch_style      = 1;
model_params.parallel_updates = 0;
model_params.dims_to_sample   = data_params.dims;
model_params.l1               = 0;  % Not implemented for everything.
model_params.l2               = 0;  % Not implemented.
if data_params.y.regression == 1
  model_params.loss_fn          = @linear_regression_sq_loss;
  model_params.gradient_fn      = @linear_regression_sq_gradient;
  model_params.curvature_fn     = @linear_regression_sq_curvature;
  model_params.hv_fn            = @linear_regression_sq_hv;
elseif data_params.y.regression == 2 && data_params.classes == 2
  model_params.loss_fn          = @binary_logistic_regression_loss;
  model_params.gradient_fn      = @binary_logistic_regression_gradient;
  model_params.curvature_fn     = @binary_logistic_regression_curvature;
  model_params.hv_fn            = @binary_logistic_regression_hv;
end

% For SGD specifically.
model_params.eta              = 1;
model_params.batch_size       = 100;
% 1=eta with no decay
% 2=eta with 1/sqrt(epoch) decay
% 3=eta with 1/epoch decay
model_params.eta_step_type    = 1;
