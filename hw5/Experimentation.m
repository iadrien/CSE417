% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)

% Problem 1: Bagged trees
load zip.train;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct1 = fitctree(X,Y);
tb1 = TreeBagger(200, X,Y, 'Method','classification');

fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct2 = fitctree(X,Y);
tb2 = TreeBagger(200, X,Y, 'Method','classification');

load zip.test;
fprintf('\nNow working on the 1v3 testing data...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
X1 = subsample(:,2:257);
Y1 = subsample(:,1);
% p1 = predict(ct1, subsample(:,2:257));
L1 = loss (ct1,X1,Y1);
LT1 = mean(error(tb1, X1, Y1));

fprintf('\nNow working on the 3v5 testing data...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
X2 = subsample(:,2:257);
Y2 = subsample(:,1);
% p2 = predict(ct2, subsample(:,2:257));
L2 = loss (ct2, X2, Y2);
LT2 = mean(error(tb2, X2,Y2));



%% Problem 2: adaboost results
load zip.train;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y131 = subsample(:,1);
X131 = subsample(:,2:257);
%ct = fitctree(X,Y,'CrossVal','on');
%fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
%bee = BaggedTrees(X, Y, 200);
%fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y351 = subsample(:,1);
X351 = subsample(:,2:257);
% ct = fitctree(X,Y,'CrossVal','on');
% fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
% bee = BaggedTrees(X, Y, 200);
% fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

load zip.test;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y132 = subsample(:,1);
X132 = subsample(:,2:257);
%ct = fitctree(X,Y,'CrossVal','on');
%fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
%bee = BaggedTrees(X, Y, 200);
%fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y352 = subsample(:,1);
X352 = subsample(:,2:257);

train13 = [];
test13 = [];
train35 = [];
test35 = [];

for numH = 1:100
    [temp11,temp12] = AdaBoost(X131,Y131,X132,Y132,numH);
    [temp21,temp22] = AdaBoost(X351,Y351,X352,Y352,numH);
    
    train13 = [train13, temp11];
    test13 = [test13,temp12];
    train35 = [train35, temp21];
    test35 = [test35,temp22];
    fprintf(int2str(numH));
end


figure, plot(train13);
hold on;
plot(test13);
figure, plot(train35);
figure, plot(test35);