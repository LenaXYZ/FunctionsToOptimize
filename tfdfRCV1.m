classdef tfdfRCV1 < SAAfunction
    
    properties
        numTrainingPoints
        shortname
        data
        numVariables
        classIndex
    end
    
    methods
        
        function thefunc = tfdfRCV1(data, shortname,classIndex)
            thefunc.data = data;
            thefunc.shortname=shortname;
            thefunc.numTrainingPoints = size(data,1);
            thefunc.numVariables=6253;
            thefunc.classIndex=classIndex;
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
            
            fx       = 0;
            grad     = zeros(size(W));
            
            % compute the function and gradient
            for i=1:BatchSize
                % get indices of nonzero elements of training point i
                tp_i=thisf.data{indices(i),1};
                % get class label
                label_i = thisf.data{indices(i),2};
                if sum( (label_i{1,1}==thisf.classIndex))==1
                    y_i=1;
                else
                    y_i=0;
                end
                
                % calculate gradient for given training pt
                if size(tp_i,1)<size(W,1)
                    tp_i(size(W,1))=0;
                end
                                
                xTw        = tp_i'*W; % x^T W
                h          = 1/(1+exp(-xTw) );
                grad = grad+(tp_i)*(h-y_i);
                
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
            
            for i=1:BatchSize
                
                % get indices of nonzero elements of training point i
                tp_i=thisf.data{indices(i),1};
                
                % calculate gradient for given training pt
                if size(tp_i,1)<size(W,1)
                    tp_i(size(W,1))=0;
                end
                xTw        = tp_i'*W; % x^T W(:,j)
                xTv        = tp_i'*V;
                h          = 1/(1+exp(-xTw) );
                hv  =  hv+h*(1-h)*xTv*(tp_i);
                
            end
            
            result.hv     = hv/BatchSize;
            result.g = NaN;
            
        end
    end
    
end

