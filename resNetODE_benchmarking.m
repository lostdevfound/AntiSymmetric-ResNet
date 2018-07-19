%
% benchmarking.m gets AntiSymResNet's performance data
%

clear;

% Load training data
load('resources/data3d/data3d.mat');
load('resources/data3d/label3d.mat');

trainLabelSet = labelSet(:,1:10000);
trainDataSet = dataSet(:,1:10000);

validLabelSet = labelSet(:,10001:30000);
validDataSet = dataSet(:,10001:30000);


load('resources/ODE_relu_net_l20_h0.5_n3_p1_s1_r0_gamma0.0001.mat')
% load('resources/Custom_relu_net_l20_h0.5_n3_p1_s1_r0_gamma0.0001.mat')

% Training and Validation costs and errors
trainingCost = 0;
generalizationCost = 0;
trainingError = 0;
validationError = 0;
upperBound = 0.95;      % def for toynet 0.95
lowerBound = 0.2;       % def for toynet 0.2
disp('benchmarking running...');


trainCycles = 10000;
validCycles = 20000;

% Training benchmarking
for i=1:trainCycles
    %  Compute training cost using 2-norm
    Y = softmax(net.forwardProp(trainDataSet(:,i)))';
    L = trainLabelSet(:,i);
    trainingCost = trainingCost + norm(Y-L)^2;
    % Compute training error
    trainingError = trainingError + computeError(Y, L, upperBound, lowerBound);
end

% Validation benchmarking
for i=1:validCycles
    % Compute validation error
    Y = softmax(net.forwardProp(validDataSet(:,i)))';
    L = validLabelSet(:,i);
    generalizationCost = generalizationCost + norm(Y-L)^2;

    validationError = validationError + computeError(Y, L, upperBound, lowerBound);

    % [resProbability,resLabel] = max(Y);
end

trainingCost = trainingCost / trainCycles;
trainingError = trainingError / trainCycles;
generalizationCost = generalizationCost / validCycles;
validationError = validationError / validCycles;

result = ['TrainingCost: ', num2str(trainingCost, '%0.2f'),' GeneralizationCost: ', num2str(generalizationCost, '%0.2f'),' TrainingError: ', num2str(trainingError,'%0.2f'), ' ValidationError: ', num2str(validationError, '%0.2f')];

disp(result);

% Binary function. If classied label does not correspond to the actual label return 1.
% If classified label is correct but classification probability is below treshold or any other classified label greater than 1 - treshold return 1
function error = computeError(predictionVec, labelVec, upperBound, lowerBound)
    [~,label] = max(labelVec);
    [predictions, labelsIndices] = sort(predictionVec, 'descend');

    if labelsIndices(1) ~= label
        error = 1.0;
%      elseif predictionVec(labelsIndices(1)) < upperBound || predictionVec(labelsIndices(2)) > lowerBound
%          error = 1.0;
    else
        error = 0.0;
    end
end

function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end

function resSoft = softmax(y_args)
    % This function computes softmax
    y_argsSum = 0;
    inputSize = max(size(y_args));

    for i = 1:inputSize
        y_argsSum = y_argsSum + exp(y_args(i));
    end

    for i = 1:inputSize
        resSoft(i) = exp(y_args(i)) / y_argsSum;
    end
end
