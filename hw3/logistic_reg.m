function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)


    [n, m] = size(X);
    iterations = 0;
    
    tolerence = 10^-3;
    while iterations < max_its
        gradient = 0;
        
        % gradient calculated from all the data points
        for index = 1:n    
            gradient = gradient + (y(index)*[1 X(index,:)])/(1+exp(y(index)*dot([1 X(index,:)], w_init)));
        end
        
        gradient = gradient / n ;
        % termination condition
        if(all(abs(gradient)<tolerence))
            break;
        end
      
        w_init = w_init + eta*gradient; %learning
        iterations = iterations + 1;
    end 
    
    w=w_init;
    e_in = 0;
    for index = 1:n
        e_in = e_in + log(1+exp(-y(index)*dot([1 X(index,:)], w_init)));
    end

    e_in=e_in/n;
    disp(iterations);

    
end

