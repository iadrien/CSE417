function [ oobErr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function

    % pre allocate bags of the tree
    bag = cell(1,numBags);
    
    dataset = [X, Y];
    [row, col] = size(X);
    
    % oob for each tree pre allocated zero
    oob = zeros(1,numBags);
    plotOOB = [];
    
    for numBag =1:numBags
          
        % with replacement but only 0.632 percent used
        sample = datasample(dataset,int32(row*0.632),'Replace',false);
            
        data = sample(:,1:col);            
        label = sample(:,col+1);
        
        % train the tree
        bag{numBag} = fitctree(data,label);        
        plabel = predict(bag{numBag},X);
        oobe = sum(Y ~= plabel);
        
        oob(numBag) = oobe;
        oobErr = sum(oob)*1.0/double(numBag*int32(row*(1-0.632)));
        plotOOB = [plotOOB, oobErr];
    end
    
    figure, plot(plotOOB);
    
end
