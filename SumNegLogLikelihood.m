classdef SumNegLogLikelihood < SAAfunction
    %SUMNEGLOGLIKELIHOOD Represent eq (4.2) in On the use of stochastic
    %hessian information
    
    properties
        numTrainingPoints
        numVariables
        data
        numFeatures
        numClasses
        shortname
        sizeNeededToRestoreGradient
    end
    
    methods
        
        function thefunc = SumNegLogLikelihood(data, shortname)
            % constructor which takes an m by n+1 matrix where each of the
            % m rows is a datapoint, the first column is a class label, and
            % the rest are features
            
            thefunc.numTrainingPoints=size(data,1); %number of training points
            thefunc.numFeatures=size(data,2)-1; %first column contains label
            thefunc.numClasses=size(unique(data(:,1)),1);
            thefunc.numVariables=thefunc.numFeatures*thefunc.numClasses; % number of variables
            thefunc.data = data;
            thefunc.shortname=shortname;
            thefunc.sizeNeededToRestoreGradient = thefunc.numClasses;
        end
        
        function result = get_f_g(varargin)
            thisf=varargin{1};
            if nargin==2
                BatchSize = thisf.numTrainingPoints;
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
            
            f=0;
            %Convert x to a matrix
            W = varargin{2};
            W=reshape(W,thisf.numFeatures,[]);
            g=zeros(size(W))          ;
            
            for k=1:BatchSize
                h=indices(k);
                % get the label
                class=thisf.data(h,1);
                % get the feature row-vector
                x=thisf.data(h,2:end);
                
                a=exp(x*W);
                b=sum(a);
                
                aDb=a/b;
                g=g+x'*aDb;
                
                g(:,class)=g(:,class)-x';
                
                f=f-log(a(class)/b);
                
            end
            
            g=reshape(g,[],1)/BatchSize; %convert to column vector
            f=f/BatchSize;
            
            result.f= f;
            result.g=g;
            if BatchSize==1
                result.compactStorage = aDb;
            end
            if nargin==3
                result.indices = indices;
            end
        end
        
        function result=get_g_hv(varargin)
            
            thisf=varargin{1};
            if nargin==3
                BatchSize = thisf.numTrainingPoints;
                indices = 1:thisf.numTrainingPoints;
            elseif nargin==4
                BatchSize = varargin{4};
                indices = randsample(thisf.numTrainingPoints,BatchSize);
            elseif nargin == 5
                BatchSize = varargin{4};
                indices = varargin{5};
                if size(indices,2)~=BatchSize
                    fprintf('size of indices must match BatchSize');
                    return;
                end
            end
            
            % V is a vector of length numfeautures*numclasses
            % W is a vector of length numfeautures*numclasses
            
            W = varargin{2};
            V = varargin{3};
            
            W=reshape(W,thisf.numFeatures,[]);
            V=reshape(V,thisf.numFeatures,[]);
            grad=zeros(size(W));  % a matrix
            hv=grad;
            
            for i=1:BatchSize
                h=indices(i);
                % get the label
                class=thisf.data(h,1);
                % get the feature row-vector
                x=thisf.data(h,2:end);
                a=exp(x*W);
                if sum(isinf(a)>=1)
                    c=find(isinf(a));
                    aDb=zeros(size(a));
                    aDb(c(1))=1;
                else
                    b=sum(a);
                    aDb=a/b;
                end
                grad=grad+x'*aDb;
                grad(:,class)=grad(:,class)-x';
                
                
                %For Hessian vector product
                
                xV=x*V;  % a matrix
                d=sum( xV.* aDb);
                
                hv=hv+x'*( aDb .* (xV-d*ones(size(a))));
                
            end
            
            grad=reshape(grad,[],1)/BatchSize;
            hv=reshape(hv,[],1)/BatchSize;
            
            result.indices = indices;
            result.hv= hv;
            result.g=grad;
            
        end
        
        function g = restoreGradient(thisf,compactGradientStorage,index)
            
            class=thisf.data(index,1);
            x=thisf.data(index,2:end);
            a=compactGradientStorage';
            b=sum(a);
            aDb=a/b;
            g=x'*aDb;
            g(:,class)=g(:,class)-x';
            g=reshape(g,[],1); %convert to column vector
            
        end
    end
    
end

