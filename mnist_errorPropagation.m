clc;
clear all;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

NN ='AntiSym'

if strcmp(NN,'AntiSym')
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/AntiSym_tan_hs_net_l10_h0.1_n40_p1_s2_r0.001_stable.mat')
elseif strcmp(NN,'ResNet')
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ResNet_tan_hs_net_l10_h0.1_n40_p1_s2_r0.001.mat')
else
    eror('Wrong NN str')
end

index = 33;
testVec1 = validatimages(:,index);
labelVec1 = validatLabels(:,index);

testVec2 = validatimages(:,index-1);
labelVec2 = validatLabels(:,index-1);

eta = 0.01; cycles = 10000;
[perturbation, perturbedVec] = PA(net, testVec1, labelVec1, eta, cycles);

disp({'perturbation norm:', num2str(norm(perturbedVec-testVec1,2))});

classifOriginal1 = net.forwardProp(testVec1);
correctPropagation1 = net.getArrayY();

classifPerturbed = net.forwardProp(perturbedVec);
fooledPropagation = net.getArrayY();

results = [ActivFunc.softmax(classifOriginal1)', ActivFunc.softmax(classifPerturbed)'];
disp('Classification Results: Original 1 Original 2 Perturbed')
disp(results);

differenceNorms = [];

% Propagation from y^2 to y^L
for i=1:net.totalNumLayers
    differenceNorms(i) = norm(correctPropagation1{i} - fooledPropagation{i});
end

disp('difference norms')
differenceNorms

% Inputs for the NN
X(1) = testVec1(1);
X(2) = perturbedVec(1);
% X(3) = testVec2(1);
Y(1) = testVec1(2);
Y(2) = perturbedVec(2);
% Y(3) = testVec2(2);
Z(1) = testVec1(3);
Z(2) = perturbedVec(3);
% Z(3) = testVec2(3);
disp('index:');disp(index);

figure
semilogy(differenceNorms)
legend('norm of error')
title({NN, 'log scale'})
xlabel('layers')
ylabel('norm of error')
