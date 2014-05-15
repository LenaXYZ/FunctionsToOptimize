classdef BadFunction < SAAfunction
    properties
        numTrainingPoints
        numVariables
        data
        shortname
        A
        B
        TAU
        GAMMA
    end
    methods
        function thefunc = BadFunction(A,B,TAU,GAMMA, shortname)
            thefunc.A=A;
            thefunc.B=B;
            thefunc.TAU=TAU;
            thefunc.GAMMA=GAMMA;
            thefunc.numTrainingPoints=size(A,1);
            thefunc.numVariables=size(A,2)*3/2;
            thefunc.shortname=shortname;
        end
        
        function result = get_f_g(varargin)
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size
            % Third argument (optional) - Indices of the batch
            
            F=varargin{1};
            if nargin==2
                indices = 1:F.numTrainingPoints;
                BatchSize = F.numTrainingPoints;
            elseif nargin==3
                BatchSize = varargin{3};
                indices = randsample(F.numTrainingPoints,BatchSize);
            elseif nargin == 4
                BatchSize = varargin{3};
                indices = varargin{4};
                if size(indices,2)~=BatchSize
                    fprintf('size of indices must match BatchSize');
                    return;
                end
            end
            
            W = varargin{2};
            
            f=0;
            
            kk3= F.numVariables/3;
            
            W1 = W(1:kk3);
            W2 = W(kk3+1:2*kk3);
            W3 = W(2*kk3+1:end);
            g1=zeros(size(W1,1),1);
            g2=zeros(size(W2,1),1);
            g3=zeros(size(W3,1),1);
            W12 = [W1;W2];
            W23 = [W2;W3];
            for i=1:BatchSize
                
                j= indices(i);
                
                f=f+exp(F.A(j,:)*W12 - F.B(j) +F.GAMMA(j) * (W23'*W23) );
                
                f=f+ F.TAU(j)*(W'*W);
                
                g1=g1+F.A(j,1:kk3)'*exp(F.A(j,:)*W12 - F.B(j)+F.GAMMA(j) *(W23'*W23) );
                
                g1=g1+ 2*F.TAU(j)*W1;
                
                g2=g2+(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*exp(F.A(j,:)*W12 - F.B(j)+F.GAMMA(j) *(W23'*W23) ) ;
                
                g2= g2 + 2*F.TAU(j)*W2;
                
                g3=g3+(2*F.GAMMA(j)*W3)*exp(F.A(j,:)*W12 - F.B(j)+F.GAMMA(j) * (W23'*W23) ) + 2*F.TAU(j)*W3;
                
                
            end
            result.g = [g1;g2;g3]/BatchSize;
            result.f =f/BatchSize;
        end
        
        function result=get_g_hv(varargin)
            
            
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size
            % Third argument (optional) - Indices of the batch
            
            F=varargin{1};
            if nargin==3
                indices = 1:F.numTrainingPoints;
                BatchSize = F.numTrainingPoints;
            elseif nargin==4
                BatchSize = varargin{4};
                indices = randsample(F.numTrainingPoints,BatchSize);
            elseif nargin == 5
                BatchSize = varargin{4};
                indices = varargin{5};
                if size(indices,2)~=BatchSize
                    fprintf('size of indices must match BatchSize');
                    return;
                end
            end
            
            W = varargin{2};
            V = varargin{3};
            
            kk3 = F.numVariables/3;
            
            W1 = W(1:kk3);
            W2 = W(kk3+1:2*kk3);
            W3 = W(2*kk3+1:end);
            g1=zeros(size(W1,1),1);
            g2=zeros(size(W2,1),1);
            g3=zeros(size(W3,1),1);
            W12 = [W1;W2];
            W23 = [W2;W3];
            h11=zeros(size(W1,1),size(W1,1));
            h12=zeros(size(W1,1),size(W1,1));
            h13 =zeros(size(W1,1),size(W1,1));
            h22 =zeros(size(W1,1),size(W1,1));
            h23 =zeros(size(W1,1),size(W1,1));
            h33 =zeros(size(W1,1),size(W1,1));
            for i=1:BatchSize
                
                j= indices(i);
                
                ee = exp(F.A(j,:)*W12 - F.B(j)+F.GAMMA(j) *(W23'*W23) );
                
                g1=g1+F.A(j,1:kk3)'*ee;
                
                g1=g1+ 2*F.TAU(j)*W1;
                
                g2=g2+(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*ee ;
                
                g2= g2 + 2*F.TAU(j)*W2;
                
                g3=g3+(2*F.GAMMA(j)*W3)*ee + 2*F.TAU(j)*W3;
                
                h11 =h11+ F.A(j,1:kk3)'*F.A(j,1:kk3) * ee + 2*F.TAU(j)*eye(kk3);
                
                
                h12 =h12+ F.A(j,1:kk3)'*(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)' * ee;
                
                h13 =h13+ F.A(j,1:kk3)'*(2*F.GAMMA(j)*W3)' * ee;
                
                h22 =h22+ (F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)'*ee + 2*eye(kk3)*(F.GAMMA(j)*ee+F.TAU(j));
                
                h23 = h23+(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*(2*F.GAMMA(j)*W3)' * ee;
                
                h33  =h33+(2*F.GAMMA(j)*W3)*(2*F.GAMMA(j)*W3)' * ee +  2*eye(kk3)*(F.GAMMA(j)*ee+F.TAU(j));
                
                
            end
            result.hv = ([h11,h12,h13;h12',h22,h23;h13',h23',h33]/BatchSize)*V;
            result.g = [g1;g2;g3]/BatchSize;
        end
        
        function result=get_hessian(varargin)
            
            
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size
            % Third argument (optional) - Indices of the batch
            
            F=varargin{1};
            if nargin==2
                indices = 1:F.numTrainingPoints;
                BatchSize = F.numTrainingPoints;
            elseif nargin==3
                BatchSize = varargin{3};
                indices = randsample(F.numTrainingPoints,BatchSize);
            elseif nargin == 4
                BatchSize = varargin{3};
                indices = varargin{4};
                if size(indices,2)~=BatchSize
                    fprintf('size of indices must match BatchSize');
                    return;
                end
            end
            
            W = varargin{2};
            
            kk3= F.numVariables/3;
            
            W1 = W(1:kk3);
            W2 = W(kk3+1:2*kk3);
            W3 = W(2*kk3+1:end);
            g1=zeros(size(W1,1),1);
            g2=zeros(size(W2,1),1);
            g3=zeros(size(W3,1),1);
            W12 = [W1;W2];
            W23 = [W2;W3];
            h11=zeros(size(W1,1),size(W1,1));
            h12=zeros(size(W1,1),size(W1,1));
            h13 =zeros(size(W1,1),size(W1,1));
            h22 =zeros(size(W1,1),size(W1,1));
            h23 =zeros(size(W1,1),size(W1,1));
            h33 =zeros(size(W1,1),size(W1,1));
            for i=1:BatchSize
                
                j= indices(i);
                
                ee = exp(F.A(j,:)*W12 - F.B(j)+F.GAMMA(j) *(W23'*W23) );
                g1=g1+F.A(j,1:kk3)'*ee;
                
                g1=g1+ 2*F.TAU(j)*W1;
                
                g2=g2+(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*ee ;
                
                g2= g2 + 2*F.TAU(j)*W2;
                
                g3=g3+(2*F.GAMMA(j)*W3)*ee + 2*F.TAU(j)*W3;
                
                h11 =h11+ F.A(j,1:kk3)'*F.A(j,1:kk3) * ee + 2*F.TAU(j)*eye(kk3);
                
                
                h12 =h12+ F.A(j,1:kk3)'*(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)' * ee;
                
                h13 =h13+ F.A(j,1:kk3)'*(2*F.GAMMA(j)*W3)' * ee;
                
                h22 =h22+ (F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)'*ee + 2*eye(kk3)*(F.GAMMA(j)*ee+F.TAU(j));
                
                h23 = h23+(F.A(j,kk3+1:end)'+2*F.GAMMA(j)*W2)*(2*F.GAMMA(j)*W3)' * ee;
                
                h33  =h33+(2*F.GAMMA(j)*W3)*(2*F.GAMMA(j)*W3)' * ee +  2*eye(kk3)*(F.GAMMA(j)*ee+F.TAU(j));
                
                
            end
            result.hessian = [h11,h12,h13;h12',h22,h23;h13',h23',h33]/BatchSize;
        end
    end
end
