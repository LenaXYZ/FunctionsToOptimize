classdef L2regFunction < SAAfunction
    
    properties
        numTrainingPoints
        shortname
        unregularizedFunction
        regParameter
        numVariables
    end
    
    methods
        
        function thefunc = L2regFunction(unregularizedFunction, regParameter)
            thefunc.unregularizedFunction = unregularizedFunction;
            thefunc.regParameter = regParameter;
            thefunc.shortname=[unregularizedFunction.shortname 'L2reg' num2str(regParameter)];
            thefunc.numTrainingPoints = unregularizedFunction.numTrainingPoints;
            thefunc.numVariables = unregularizedFunction.numVariables;
        end
        function result = get_f_g(varargin)
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size
            % Third argument (optional) - Indices of the batch
          
            thisf=varargin{1};
            
            resultU = thisf.unregularizedFunction.get_f_g(varargin{2:end});
           

            W = varargin{2};
            
            result.f = resultU.f + (1/2)*thisf.regParameter*(W'*W);
            result.g = resultU.g + thisf.regParameter*W;
            
        end
        
        function result=get_g_hv(varargin)
            
            thisf=varargin{1};
            
            resultU  = thisf.unregularizedFunction.get_g_hv(varargin{2:end});
            
            W = varargin{2};
            V = varargin{3};
            
            result.g = resultU.g + thisf.regParameter*W;
            result.hv = resultU.hv + thisf.regParameter*V;
        end
        
        
        function result=get_hessian(varargin)
            
            thisf=varargin{1};
            
            resultU  = thisf.unregularizedFunction.get_hessian(varargin{2:end});
            
            result.hessian = resultU.hessian + thisf.regParameter*eye(thisf.numVariables);
        end
    end
    
end

