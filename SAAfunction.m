classdef SAAfunction
    %SAAFUNCTION Abstract class that represents an SAA function
    % It has a form 1/m \sum_{h=1}^m f_h(x)
    % All f_h come from the same distribution
    
    properties (Abstract)
        numTrainingPoints % Number of terms in the summation
        shortname         % String representation of the problem
        numVariables     
    end
    
    methods (Abstract)
        
        result = get_f_g(varargin);
            % First argument - the point at which to evaluate function and
            % gradient
            % Second argument (optional) - Batch size            
            % Third argument (optional) - Indices of the batch     
            
            % result.f - the function value
            % result.g - the gradient 
            
        result = get_g_hv(varargin);
    end
    
end

