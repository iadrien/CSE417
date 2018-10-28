training = csvread('clevelandtrain.csv',1);

[row, col] = size(training);

trainingData = [];
trainingLabel = [];

for index = 1:row
    trainingData = [trainingData; training(index,1:col-1)];
    trainingLabel = [trainingLabel; training(index,col)*2-1];
end

test = csvread('clevelandtest.csv',1);

[row, col] = size(test);

testData = [];
testLabel = [];

for index = 1:row
    testData = [testData; test(index,1:col-1)];
    testLabel = [testLabel; test(index,col)*2-1];
end

mdl = glmfit(trainingData,trainingLabel);
classificationtrain = find_test_error(mdl,trainingData,trainingLabel);
classification = find_test_error(mdl,testData,testLabel);

disp(classification);
disp(classificationtrain);
