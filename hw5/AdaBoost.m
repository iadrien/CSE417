function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

    % pre allocate weights
    w = ones(size(x_tr,1),1)./size(x_tr,1);

    % learn the hypothesis
    for i = 1:n_trees
        MDL = fitctree(X_tr,y_tr,'SplitCriterion','deviance',w);

        % prediction on current model
        prediction = predict(MDL,X_tr);
        epsilon = sum(w .* (predict(MDL,X_tr)~= y_tr));

        Z = 2*sqrt(epsilon*(1 - epsilon));
        w = w .*(exp(-0.5*log((1 - epsilon)/epsilon) .*y_tr.* transpose(prediction)))/Z;

    end
    
    % final classifier
    tr = templateTree('SplitCriterion','deviance');   
    MDL = fitensemble(X_tr, y_tr, 'AdaBoostM1',n_trees,tr);

    % error measures
    train_err = resubLoss(MDL);
    test_err = loss(MDL, X_te, y_te);

end

