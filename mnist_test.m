clear;
clc;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

% Setup NN's params
h = 0.2;        % default 0.7
igamma = 0.02;       % default 0.1
trainCycles = 400000;       % default 400000
eta = 0.0035;           % good default 0.0005 or 0.003
initScaler = 0.07;
neurons = 50;
layers = 10;

% Define different params for several NNs
numNets = 13;
listLayers =  [2,    2,   3,   3,    5,    10,  10,  10,   10,  20,   20,   30,  30];
listH =       [0.8, .35,  .3,  .8,   .25,  .2,  .6,  .8,   1,   .4,   .8,   .05,  0.1];
listNeurons = [100,  100, 100, 100,  50,   20,  20,  20,   20,  20,   20,   15,  15];

% Set to true if need to retrain
first_time_launch = false;
doPerturbation = true;

% Create multiple neural nets with different params
for i=13:numNets

    layers = listLayers(i);
    neurons = listNeurons(i);
    h = listH(i);

    % Training part.
    if first_time_launch == true

        % Init NN and train it
        net = AntiSymResNet(layers, 784, 10, neurons, igamma, h, initScaler, false);
        disp('training...');
        net.train(trainImages, trainLabels, trainCycles, eta);
        disp('training complete.');

        % Save trained net
        netStr = {'net_l', num2str(layers), '_h', num2str(h), '_ig', num2str(igamma), '_n', num2str(neurons), '.mat','resources/'};
        str = strcat(netStr{10},netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9});
        net.name = strcat(netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9});
        save(str,'net');
    else
        load('resources/net_l30_h0.05_ig0.02_n15.mat');    % Load pretrained AntiSymResNet
    end
end

% Pick image then forwardProp image and print result in the console.
index = 45;     % Pick some image by its index (digit 3 is index 33)
testImg =  validatimages(:,index);
[~,digitNumber] = max(validatLabels(:,index))
perturbedImg = testImg;
classifRes = ones(10,1);

noisyImg = min(testImg + 0.5*rand(784,1), 1);   % limit the range from 0 to 1

% Perturbation generation
disp('working...');
while classifRes(digitNumber) > 0 && doPerturbation == true
    net.forwardProp(perturbedImg);
    perturbedImg = net.adversBackProp(perturbedImg,validatLabels(:,index), 0.9);
    classifRes = sigm(net.forwardProp(perturbedImg));

    [prediction,maxInd] = max(classifRes);
    % Break if correct classif index is not max
    if maxInd ~= digitNumber
        break;
    end
end
normOfPerturbation = norm(perturbedImg - testImg)

% Classify images
classificationResultPerturb = sigm(net.forwardProp(perturbedImg));
classificationResultOrig = sigm(net.forwardProp(testImg));
classificationResultNoisy = sigm(net.forwardProp(noisyImg));
results = [classificationResultPerturb, classificationResultOrig,classificationResultNoisy]

% Didsplay picked image
figure;
subplot(1,3,1);
digitPerturbed = reshape(perturbedImg, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');

subplot(1,3,2);
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])      % show the image
title('original');

subplot(1,3,3);
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');



function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end
