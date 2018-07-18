% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
h = 1;                          % default 0.7
igamma = 0;                     % default 0.1
trainCycles = 400000;           % default 400000
eta = 0.002;                    % good default 0.0005 or 0.003
initScaler = 0.1;               % default 0.01
n = 20;                         %neurons
layers = 2;
activ = 'relu';
p = 5;
s = 2;
r = 0.003;

first_time_launch = true;
doPerturbation = true;
%                                           %
%              Training part                %
%                                           %

if first_time_launch == true
    % Init NN and train it
    net = ResNetCustom(layers, 784, 10, n, igamma, h, initScaler, false, activ, p, s, r);
    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/','custom_',  activ, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(n), '_p', num2str(p), '_s', num2str(s), '_r', num2str(r),'.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');
else
    load('resources/sqrf_net_l2_h1_n20_r0.003.mat');    % Load pretrained AntiSymResNet
end



%                                           %
%              Perturbation part            %
%                                           %
normSum = 0;
samples = 30;
offset=1;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testImg =  validatimages(:,index);
    [~,digitNumber] = max(validatLabels(:,index));

    perturbedImg = testImg;
    classifRes = ones(10,1);

    noisyImg = min(testImg + 0.5*rand(784,1), 1);   % limit the range from 0 to 1

    % Perturbation generation
    count = 0;
    breakCount = 10000;

    while classifRes(digitNumber) > 0 && doPerturbation == true
        net.forwardProp(perturbedImg);
        perturbedImg = net.adversBackProp(perturbedImg,validatLabels(:,index), 0.02);   % def 0.02
        classifRes = softmax(net.forwardProp(perturbedImg));

        [prediction,maxInd] = max(classifRes);

        % Break if correct classif index is not max
        if maxInd ~= digitNumber
            disp('fooled');
            break;

        elseif count > breakCount
            perturbedImg = testImg;
            samples = samples -1;
            disp('breaking...........');disp('sample number:');disp(k);
            [classifRes',validatLabels(:,index)]
            break;
        end

        count = count + 1;
    end
%     absNormOfPerturbation = norm(perturbedImg - testImg)
    relNormOfPerturbation = norm(perturbedImg - testImg)/norm(testImg);
    normSum = normSum + relNormOfPerturbation;
end

RelativeAvgNorm = normSum/(samples+1)

% Classify images
classificationResultPerturb = softmax(net.forwardProp(perturbedImg))';
classificationResultOrig = softmax(net.forwardProp(testImg))';
classificationResultNoisy = softmax(net.forwardProp(noisyImg))';
results = [classificationResultPerturb, classificationResultOrig, classificationResultNoisy]

% Didsplay picked image
figure;
subplot(1,3,1);
perturbedImg = perturbedImg + validMean;
digitPerturbed = reshape(perturbedImg, [28,28]);    % row = 28 x 28 image
imshow(digitPerturbed*255, [0 255])      % show the image
title('perturbed');

subplot(1,3,2);
testImg = testImg + validMean;
digitOrig = reshape(testImg, [28,28]);    % row = 28 x 28 image
imshow(digitOrig*255,[0 255])      % show the image
title('original');

subplot(1,3,3);
noisyImg = noisyImg + validMean;
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');



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
