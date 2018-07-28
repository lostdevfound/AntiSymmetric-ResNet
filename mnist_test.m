% clc;
clear;
% Load training data
[trainImages,trainLabels, validatimages, validatLabels] = loadMNIST('mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte','mnist/t10k-images.idx3-ubyte','mnist/t10k-labels.idx1-ubyte');

trainMean = mean(trainImages(:,1:end),2);
validMean = mean(validatimages(:,1:end),2);
trainImages = trainImages - trainMean;
validatimages = validatimages - validMean;

% Setup NN's params
NN = 'ResNet';        % Train 'AntiSym' or 'ResNet'
trainCycles = 500000;        % default 400000
eta = 0.001;                 % good default 0.0005 or 0.003
neurons = 40;
layers = 10;
initScaler = 0.01;           % default 0.01
h = 0.1;                     % default 0.1
igamma = 0.01;               % default 0.01
regular = 0.001;                 % regularization
activ = 'tan_hs';
p = 1;
s = 2;

first_time_launch = false;
doPerturbation = true;


%                                           %
%              Training part                %
%                                           %

if first_time_launch == true

    % Init NN and train it
    if strcmp(NN, 'AntiSym')
        net = ResNetAntiSym(layers, 784, 10, neurons, igamma, h, initScaler, false, activ, p, s, regular);
    elseif strcmp(NN, 'ResNet')
        net = ResNetCustom(layers, 784, 10, neurons, h, initScaler, false, activ, p, s, regular);
    else
        error('Wrong NN string');
    end

    disp('training...');
    net.train(trainImages, trainLabels, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', NN, '_',  activ, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(neurons), '_p', num2str(p), '_s', num2str(s), '_r', num2str(regular),'.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');
else
%     load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/AntiSymResNetrelu_net_l5_h0.4_n70_p1_s1_r0.mat')
%     load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/ResNet_relu_net_l5_h0.4_n70_p1_s1_r0.mat')
load('/home/user1/Documents/ML/matlab/AntiSymResNet/resources/interpolated.mat')

end



%                                           %
%              Perturbation part            %
%                                           %
normSum = 0;
samples = 100;
offset=1;
pert_eta = 0.01;
pertCycles = 10000;
perturbedImg = 0;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testImg =  validatimages(:,index);
    labelVec = validatLabels(:,index);

    noisyImg = min(testImg + 0.3*rand(784,1), 1);   % limit the range from 0 to 1

    [peturbation, perturbedImg] = PA(net, testImg, labelVec, pert_eta, pertCycles);

    relNormOfPerturbation = norm(perturbedImg - testImg)/norm(testImg);
    normSum = normSum + relNormOfPerturbation;
end
RelativeAvgNorm = normSum/(samples+1);
disp({'RelativeAvgNorm:', num2str(RelativeAvgNorm)});


% Classify images
classificationResultPerturb = ActivFunc.softmax(net.forwardProp(perturbedImg))';
classificationResultOrig = ActivFunc.softmax(net.forwardProp(testImg))';
classificationResultNoisy = ActivFunc.softmax(net.forwardProp(noisyImg))';
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
imshow(digitOrig*255,[0 255])             % show the image
title('original');

subplot(1,3,3);
noisyImg = noisyImg + validMean;
digitNoise = reshape(noisyImg, [28,28]);    % row = 28 x 28 image
imshow(digitNoise*255,[0 255])      % show the image
title('random noise');
