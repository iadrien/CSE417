training = csvread('clevelandtrain.csv',1);

[row, col] = size(training);

trainingData = [];
trainingLabel = [];

for index = 1:row
    trainingData = [trainingData; training(index,1:col-1)];
    trainingLabel = [trainingLabel; training(index,col)*2-1];
end

%trainingData = zscore(trainingData,1,1);
initial_weights = zeros(1, col);

[w, ein]=logistic_reg( trainingData, trainingLabel, initial_weights, 10000, 10^-5);

test = csvread('clevelandtest.csv',1);

[row, col] = size(test);

testData = [];
testLabel = [];

for index = 1:row
    testData = [testData; test(index,1:col-1)];
    testLabel = [testLabel; test(index,col)*2-1];
end

classification = find_test_error(w,testData,testLabel);
classificationTrain = find_test_error(w,trainingData,trainingLabel);
disp(ein);
disp(classification);
disp(classificationTrain)