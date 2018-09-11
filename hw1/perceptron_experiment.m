function [ num_iters bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bound_minus_ni is the difference between the theoretical bound
%               and the actual number of iterations
%      (both the outputs should be num_samples long)

    data = [];
    dataWOLabel = [];
    num_iters = [];
    p = 0;
    r = 0;
    bounds = [];
    
%for each itereation, an experiment of 100 sample is performed
%for convenience, w is first set to 10 dimension vector 
    for i=1:num_samples
        w = rand(1,d);
    
        for j=1:N
            x = rand(1,d)*2-1;
            y = sign(dot(w,x));
            temp = [x y];
            data = [data; temp];
            dataWOLabel = [dataWOLabel; x];
        end
        
        [weightVector, iteration] = perceptron_learn(data);
        p = min(abs(weightVector*transpose(dataWOLabel)));
        r = norm(norm(dataWOLabel,Inf));
        
        num_iters = [num_iters iteration];
        data = [];
        dataWOLabel = [];
        
        bound = (r * norm(weightVector)/ p).^2;
        bounds = [bounds bound];
    end
    figure;
    logdif = log(bounds-num_iters);
    histogram(logdif);
    figure;
    histogram(num_iters);
end

