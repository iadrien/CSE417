function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    w=[0 0 0 0 0 0 0 0 0 0 ];
    iterations = 0;
    

    while true   
        mc = missClassified(data_in,w);
        
        if size(mc)==0
            break;
        end
        [r1, r2]=size(mc);
        index = randi([1 r1],1,1);

        x = mc(index,:);
        y = x(r2);
        wx = sign(dot(w,x(1:r2-1)));
        
        w = w + y*x(1:r2-1);
        iterations = iterations + 1;
    end
end

function [missSet] = missClassified (data_in, w)
    missSet = [];
    [m, n]=size(data_in);

    for index = 1:m   
        x = data_in(index,:);
        y = x(n);
        wx = sign(dot(w,x(1:n-1)));  
        if wx ~= y
            missSet=[missSet; x];
        end
    end
end

