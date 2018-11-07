function [ overfit_m ] = computeOverfitMeasure( true_Q_f, N_train, N_test, var, num_expts )
%COMPUTEOVERFITMEASURE Compute how much worse H_10 is compared with H_2 in
%terms of test error. Negative number means it's better.
%   Inputs
%       true_Q_f: order of the true hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       var: variance of the stochastic noise
%       num_expts: number of times to run the experiment
%   Output
%       overfit_m: vector of length num_expts, reporting each of the
%                  differences in error between H_10 and H_2

Eout_g2 = [];
Eout_g10 = [];
for index = 1:num_expts
    [train_set, test_set]=generate_dataset(true_Q_f,N_train,N_test, var^0.5);
    %X = train(:,1);
    %trainMatrix2 = computeLegPoly(transpose(X),2);
    glm2 = glmfit(computeLegPoly(train_set(:,1),2),train_set(:,2),'normal','constant','off');
    %trainMatrix10=computeLegPoly(transpose(X),10);
    glm10 = glmfit(computeLegPoly(train_set(:,1),10),train_set(:,2),'normal','constant','off');
    
    g_2 = computeLegPoly(test_set(:,1),2)*glm2;
    g_10 = computeLegPoly(test_set(:,1),10)*glm10;
    Eout_g2  = [Eout_g2, mean((g_2 - test_set(:,2)).^2)];
    Eout_g10 = [Eout_g10, mean((g_10 - test_set(:,2)).^2)];
end

overfit_m = [mean(Eout_g10) - mean(Eout_g2);median(Eout_g10 - Eout_g2)];

end