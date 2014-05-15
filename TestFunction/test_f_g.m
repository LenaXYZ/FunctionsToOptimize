function test_f_g(thefunc,x,testIndex)
% Tests whether the get_f_g works correctly for a given function
% Test 1: Check gradient by finite differences
% Test 2: Check sampling
% Test 3: Check deterministic sampling

switch testIndex
    case 1
        fprintf('In this test, if relerror is small, gradient and f computations are compatible\n');
        startpow = 8;
        result = thefunc.get_f_g(x);
        trueGradient = result.g;
        trypowforcase1(thefunc,x,startpow,0,0,trueGradient);
    case 2
        fprintf('In this test, if relerror is decreasing, sampling is working \n');
        result = thefunc.get_f_g(x);
        trueGradient = result.g;
        numDiffSamplesizesToTry = 100;
        samplesizes = round(logspace(log10(1),log10(thefunc.numTrainingPoints),numDiffSamplesizesToTry));
        j=0;
        for i = 1:size(samplesizes,2)
            if i>1 && samplesizes(i)==samplesizes(i-1)
                continue;
            end
            j=j+1;
            result = thefunc.get_f_g(x, samplesizes(i));
            fprintf('For sample size %8i',samplesizes(i));
            fprintf('    the relative error is %13.3e\n', norm(result.g -trueGradient )/norm(trueGradient));
            relerrors(j) =norm(result.g -trueGradient )/norm(trueGradient);
            samplesizesforlabel{j} = num2str(samplesizes(i));
        end
        figure
        plot(relerrors)
        ylabel('relative error');
        xlabel('batchSize');
        numticks = 10;
        skipevery = floor(size(relerrors,2)/numticks);
        set(gca,'XTick',1:skipevery:size(relerrors,2))
        set(gca,'XTickLabel',samplesizesforlabel(1:skipevery:size(relerrors,2)))
    case 3
        fprintf('In this test, if differnce is small, determinitstic sampling is working \n');
        result = thefunc.get_f_g(x);
        trueGradient = result.g;
        gradientEstimate = zeros(size(x));
        for i=1:thefunc.numTrainingPoints
            result = thefunc.get_f_g(x,1,i);
            gradientEstimate=gradientEstimate+result.g/thefunc.numTrainingPoints;
        end
        fprintf('The relative error is %13.3e\n', norm(gradientEstimate -trueGradient )/norm(trueGradient));
    case 4
        fprintf('Test hessian vector computation. Tiny numbers are good.\n ')
        vecc = randn(thefunc.numVariables,1);
       
        
        res1 = thefunc.get_g_hv(x,vecc);
        
        estimatedHV = res1.hv;
        
        yo = thefunc.get_f_g(x );
        
        gradientAtX = yo.g;
        
        epsilons = [1e-10;1e-9;1e-8;1e-7;1e-6;1e-5;1e-4;1e-3;1e-2];
        
        for i=1:size(epsilons,1)
            epsilon=epsilons(i);
            
           
            
            displaced = thefunc.get_f_g(x + epsilon* vecc);
            res1 = (displaced.g -gradientAtX)/epsilon;
            errrorrr  = norm(res1 -estimatedHV )/norm(estimatedHV);
            fprintf('epsilon = %i  error = %i\n', epsilon, errrorrr);
        end
end
end

function trypowforcase1(thefunc,x,pow,direction, prevdifference,trueGradient)
epsilon=10^(-pow);
gradestimate = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        vectorWith1 = zeros(size(x));
        vectorWith1(i,j)=1;
        forwardjumpresult = thefunc.get_f_g(x+epsilon*vectorWith1);
        backwardjumpresult = thefunc.get_f_g(x-epsilon*vectorWith1);
        gradestimate(i,j)=(forwardjumpresult.f - backwardjumpresult.f)/(2*epsilon);
    end
end
difference=norm(gradestimate - trueGradient);
fprintf('For epsilon %13.3e',epsilon);
fprintf('    the relerror is %13.3e\n', norm(difference)/norm(trueGradient));
if isnan(difference)
elseif prevdifference>0.9*difference && direction==1
    trypowforcase1(thefunc,x,pow+1,direction, difference,trueGradient)
elseif prevdifference>0.9*difference && direction==-1
    trypowforcase1(thefunc,x,pow-1,direction, difference,trueGradient)
end
if direction==0
    trypowforcase1(thefunc,x,pow-1,-1, difference,trueGradient)
    trypowforcase1(thefunc,x,pow+1,1, difference,trueGradient)
end
end