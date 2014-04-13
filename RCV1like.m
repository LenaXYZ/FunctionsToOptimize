classdef RCV1like < SAAfunction
    
    properties
        numTrainingPoints
        shortname
        data
        numVariables
    end
    
    methods
        
        function thefunc = RCV1like(data, shortname)
            thefunc.data = data;
            thefunc.shortname=shortname;
            thefunc.numTrainingPoints = size(data,1);
            thefunc.numVariables=length(data{1,1});
        end
        function result = get_f_g(varargin)
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size
            % Third argument (optional) - Indices of the batch
            
            thisf=varargin{1};
            if nargin==2
                indices = 1:thisf.numTrainingPoints;
                BatchSize= size(indices,2);
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
            
            
            ge       = 0;
            fx       = 0;
            grad     = zeros(size(W));
            vec_ones = ones(112919+1,1);
            
            % compute the function and gradient
            for i=1:BatchSize
                % get indices of nonzero elements of training point i
                tp_i=thisf.data{indices(i),1};
                % get class label
                label_i = thisf.data{indices(i),2};
                if sum( (label_i{1,1}==4))==1
                    y_i=1;
                else
                    y_i=0;
                end
                
                % calculate gradient for given training pt
                xTw        = sum( W(tp_i)); % x^T W
                h          = 1/(1+exp(-xTw) );
                grad(tp_i) = grad(tp_i)+vec_ones(tp_i)*(h-y_i);
                
                % calculate fx for given training pt
                if h==1
                    fx = fx - y_i*log(1);
                elseif h==0
                    fx = fx -(1-y_i)*log(1);
                else
                    fx = fx -( y_i*log(h) + (1-y_i)*log(1-h));
                end
            end
            result.g = grad/BatchSize;
            result.f   = fx/BatchSize;
        end
        
        function result=get_g_hv(varargin)
            
            thisf=varargin{1};
            W = varargin{2};
            V = varargin{3};
            BatchSize = varargin{4};
            
            indices = randsample(thisf.numTrainingPoints,BatchSize);
            
            hv       = zeros(size(W));
            vec_ones = ones(112919+1,1);
            
            for i=1:BatchSize
                
                % get indices of nonzero elements of training point i
                tp_i=thisf.data{indices(i),1};
                
                % calculate gradient for given training pt
                xTw        = sum( W(tp_i)); % x^T W(:,j)
                xTv        = sum( V(tp_i));
                h          = 1/(1+exp(-xTw) );
                hv(tp_i)   =  hv(tp_i)+h*(1-h)*xTv*vec_ones(tp_i);
                
            end
            
            result.hv     = hv/BatchSize;
            result.g = NaN;
            
        end
    end
    
end

