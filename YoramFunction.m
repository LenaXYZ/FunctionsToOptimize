classdef YoramFunction < SAAfunction    
    properties
        numTrainingPoints
        numVariables
        data
        shortname
        model_params_yoram
    end
    methods        
        function thefunc = YoramFunction(data,model_params, shortname)
            thefunc.data=data;
            thefunc.model_params_yoram = model_params;
            thefunc.numTrainingPoints=size(thefunc.data.X,1);
            thefunc.numVariables=size(thefunc.data.X,2);
            thefunc.shortname=shortname;
        end
        
        function result = get_f_g(varargin)
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size            
            % Third argument (optional) - Indices of the batch
                        
            thisf=varargin{1};
            if nargin==2
                indices = 1:thisf.numTrainingPoints;
            elseif nargin==3
                BatchSize = varargin{3};
                indices = randsample(thisf.numTrainingPoints,BatchSize);
            elseif nargin == 4
                BatchSize = varargin{3};
                indices = varargin{4};
                if size(indices,2)~=BatchSize
                    fprintf('size of indices must match BatchSize');
                    return;
                end
            end
            
            W = varargin{2};
            
            result.g = thisf.model_params_yoram.gradient_fn(W, thisf.data.X(indices,:), thisf.data.Y(:,indices))';
            result.f = thisf.model_params_yoram.loss_fn(W, thisf.data.X(indices,:), thisf.data.Y(:,indices));
        end
        
        function result=get_g_hv(varargin)
                        
            thisf=varargin{1};
            
            W = varargin{2};
            V = varargin{3};
            
            BatchSize = varargin{4};
            
            indices = randsample(thisf.numTrainingPoints,BatchSize);
            
            result.g  = thisf.model_params_yoram.gradient_fn(W, thisf.data.X(indices,:), thisf.data.Y(:,indices))';
            result.hv = thisf.model_params_yoram.hv_fn(W, thisf.data.X(indices,:), thisf.data.Y(:,indices),V);
                    
        end
        
        function result=get_hessian(varargin)
                        
            thisf=varargin{1};
            
            W = varargin{2};
            
            if nargin==2
                indices = 1:thisf.numTrainingPoints;
            elseif nargin==3           
                BatchSize = varargin{3};
                indices = randsample(thisf.numTrainingPoints,BatchSize);  
            end            
            
            result.hessian = thisf.model_params_yoram.hessian_fn(W, thisf.data.X(indices,:), thisf.data.Y(:,indices));
            
        end
    end
end
