clc;
clear all;
close all
% Training dataset: 
NTrain = 1000;
NValid = 100;
trainDataSet  = 2*rand(2, NTrain)-1;

%trainLabelSet(1,:) = (trainDataSet(1,:)+1).^2 +  (trainDataSet(2,:)+1).^2 <= 2^2; % quarter of cirle
trainLabelSet(1,:) = trainDataSet(1,:) <=  trainDataSet(2,:); % linear region
trainLabelSet(2,:) = 1-trainLabelSet(1,:);

scatter(trainDataSet(1,:),trainDataSet(2,:),5*ones(1,NTrain),trainLabelSet(1,:))
title('Training Data')
colormap winter



% Setup NN's params
NN_type = 'ODE';          % ODE or Custom where Custom is a regular ResNet
igamma = 1e-7;          % default 0.0001
trainCycles = 200000;       % default 400000
eta = 0.1;                % good default 0.003
initScaler = 1;           % default 0.01
neurons = 2;
layers = 5;
h = 0.1;
activFunc = 'relu';
regular = 0;
p = 1; % activation function parameter
s = 1; % activation function parameter


% Set to true if need to retrain
first_time_launch = false;
doPerturbation = false;

%                                           %
%              Training part                %
%                                           %

if first_time_launch == true
    % Init NN and train it. Params i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode
    if strcmp(NN_type, 'ODE')
        disp('training...'); disp(NN_type);
        net = ResNetAntiSymODE(layers, neurons, neurons, neurons, igamma, h, initScaler, false, activFunc, regular, p, s);
    elseif strcmp(NN_type, 'Custom')
        disp('training...'); disp(NN_type);
        net = ResNetCustom(layers, neurons, neurons, neurons, h, initScaler, false, activFunc, p, s, regular);
    end

    net.train(trainDataSet, trainLabelSet, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', NN_type, '_',activFunc, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(neurons), '_p', num2str(p), '_s', num2str(s), '_r', num2str(regular),'_gamma',num2str(igamma),'_2d_',date,'.mat'};
    [~,numNames] = size(netStr);
    str = '';

    for i=1:numNames
        str = strcat(str,netStr(i));
    end
    save(str{1},'net');

else

    load('resources/ODE_relu_net_l15_h0.1_n3_p1_s1_r0_gamma0.0001.mat')
%     load('resources/Custom_relu_net_l10_h0.5_n3_p1_s1_r0_gamma0.0001.mat')
end



