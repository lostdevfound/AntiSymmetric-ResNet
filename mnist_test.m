clear;
clc;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

% Setup NN's params
h = 0.4;        % default 0.7
igamma = 0.1;       % default 0.1
trainCycles = 850000;       % default 400000
eta = 0.0004;           % good default 0.0005
initScaler = 0.07;

% load('resources/net_20_layer_h01_igamma03');    % Load pretrained AntiSymResNet

% Set to true if need to retrain
first_time_launch = true;
doPerturbation = true;


% Training part.
if first_time_launch == true
    % Init NN and train it
    net = AntiSymResNet(20, 784, 10, 15, igamma, h, initScaler, false);
    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);      % fast but not accurate training
    disp('training complete.');
end

save 'net_20_layer_h04_igamma01.mat' net;

% Pick image then forwardProp image and print result in the console.
index = 99;     % Pick some image by its index (digit 3 is index 33)
testImg =  validatimages(:,index);
[~,digitNumber] = max(validatLabels(:,index))
perturbedImg = testImg;
classifRes = ones(10,1);

noisyImg = min(testImg + 0.3*rand(784,1), 1);   % limit the range from 0 to 1

% Perturbation generation
disp('working...');
while classifRes(digitNumber) > 0.5 && doPerturbation == true
% for i=1:500000
    net.forwardProp(perturbedImg);
    perturbedImg = net.adversBackProp(perturbedImg,validatLabels(:,index), 0.1);
    classifRes = sigm(net.forwardProp(perturbedImg));
    classifRes(digitNumber);
end

% Classify images
classificationResultPerturb = sigm(net.forwardProp(perturbedImg))
classificationResultOrig = sigm(net.forwardProp(testImg))
classificationResultNoisy = sigm(net.forwardProp(noisyImg))
normOfPerturbation = norm(perturbedImg - testImg)

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
norm(perturbedImg -testImg)

subplot(1,3,3);
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');



function y = sigm(z)
    % sigmoid activation function.
    y = 1./(1+exp(-z));
end
