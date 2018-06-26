% clear;
clc;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

% Setup NN's params
h = 0.8;
igamma = 0.01;
trainCycles = 400000;
eta = 5;
initScaler = 0.5;

% Set to true if need to retrain
first_time_launch = true;

% Training part.
if first_time_launch == true
    % Init NN and train it
    net = AntiSymResNet(2, 784, 10, 20, igamma, h, initScaler, false);
    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);      % fast but not accurate training
    disp('training complete.');
end

index = 33;     % Pick some image by its index (digit 3 is index 33)
testImg =  validatimages(:,index);

classificationResultOrig = net.forwardProp(testImg)

figure();
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])      % show the image
title('original');
