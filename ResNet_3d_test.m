clc;
clear all;
% Load training data
load('resources/data3d/data3d.mat');
load('resources/data3d/label3d.mat');
trainLabelSet = labelSet(:,1:10000);
trainDataSet = dataSet(:,1:10000);

validLabelSet = labelSet(:,10001:30000);
validDataSet  = dataSet(:,10001:30000);

% Setup NN's params
NN_type = 'ODE';          % ODE or Custom where Custom is a regular ResNet
igamma = 0.0001;          % default 0.0001
trainCycles = 600000;       % default 400000
eta = 0.01;                % good default 0.003
initScaler = 0.5;           % default 0.01
neurons = 3;
layers = 15;
h = 0.3;
activFunc = 'relu';
regular = 0.0001;
p = 1; % activation function parameter
s = 1; % activation function parameter


% Set to true if need to retrain
first_time_launch = true;
doPerturbation = false;

%                                           %
%              Training part                %
%                                           %

if first_time_launch == true
    % Init NN and train it. Params i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode
    if strcmp(NN_type, 'ODE')
        disp('training...'); disp(NN_type);
        net = ResNetAntiSymODE(layers, 3, 3, neurons, igamma, h, initScaler, false, activFunc, regular, p, s);
    elseif strcmp(NN_type, 'Custom')
        disp('training...'); disp(NN_type);
        net = ResNetCustom(layers, 3, 3, neurons, h, initScaler, false, activFunc, p, s, regular);
    end

    net.train(trainDataSet, trainLabelSet, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', NN_type, '_',activFunc, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(neurons), '_p', num2str(p), '_s', num2str(s), '_r', num2str(regular),'_gamma',num2str(igamma), '.mat'};
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



%                                           %
%              Perturbation part            %
%                                           %

normSum = 0;
samples = 1; % number of vector being perturbed
offset= 4;
label = 0;
pert_eta = 0.01;
pertCycles = 10000;
foolCount = 0;
onenormVecNum = 0;

for k = offset:offset + samples
    index = k;     % Pick some image by its index
    testVec =  validDataSet(:,index);
    labelVec =  validLabelSet(:,index);

    [~,labelIndex] = max(labelVec);
    [~, predictionInd] = max(net.forwardProp(testVec));

    % Skip vectors that lie outside the sphere or if prediction is not correct
    if labelIndex == 2 || predictionInd ~= labelIndex
        disp('skipping');
        continue;
    else
        onenormVecNum = onenormVecNum + 1;
    end


    % Perturbation generation
    perturbedVec = 0;
    if doPerturbation == true
        [peturbation, perturbedVec] = PA(net, testVec, labelVec, pert_eta, pertCycles);
    end


    % Get classification result for pertrubed vector
    [~, predictionInd] = max(net.forwardProp(perturbedVec));

    % Count the nunmber of vecs that have norm < 1 and fooled the NN
    if  norm(perturbedVec) ~= 0 && (predictionInd ~= labelIndex) && norm(perturbedVec) < 1 && norm(testVec) < 1
        foolCount = foolCount + 1;
        disp('fooled count:'); disp(foolCount);
        relNormOfPerturbation = norm(perturbedVec - testVec)/norm(testVec);
        normSum = normSum + relNormOfPerturbation;

        X(1) = testVec(1);
        X(2) = perturbedVec(1);
        Y(1) = testVec(2);
        Y(2) = perturbedVec(2);
        Z(1) = testVec(3);
        Z(2) = perturbedVec(3);
        disp('index:');disp(index);
        figure
        [xs,ys,zs] = sphere;
        h = surfl(xs, ys, zs);

        colormap([0 0 0]);
        set(h, 'FaceAlpha', 0.1);
        shading interp;
        hold on;
        scatter3(X,Y,Z, [100 100], [0 0 1; 1 0 0], '*');
        set(gca, 'Projection','perspective');
        legendStrOrig = {'original','\index:', num2str(index), ', vec norm:', num2str(norm(testVec))};
        legendStrOrig = strcat(legendStrOrig{1:end});
        legendStrPerturbed = {'perturbed ', ', vec norm:', num2str(norm(perturbedVec))};
        legendStrPerturbed = strcat(legendStrPerturbed{1:end});
        legScatter = legend({legendStrOrig, legendStrPerturbed}, 'FontSize', 12);
        leg_pos = get(legScatter,'position');
        set(legScatter,'position',[leg_pos(1),leg_pos(2),leg_pos(3)*1.2,leg_pos(4)*2]);

    end
    label = validLabelSet(:,index);

end

% Classify images
classificationResultPerturb = ActivFunc.softmax(net.forwardProp(perturbedVec))';
classificationResultOrig = ActivFunc.softmax(net.forwardProp(testVec))';
disp('results: label, classifPerturbed, classifOrig');
results = [label,classificationResultPerturb, classificationResultOrig]
disp('norms: perturbed norm, orig norm')
norms = [norm(perturbedVec), norm(testVec)]

disp('### Overall report ###');
disp('Vectors sampled:');
disp(onenormVecNum);
fooledPercent = foolCount/onenormVecNum*100;
disp('Percent fooled:');
disp(fooledPercent);
disp('relativeAvgNorm:')
relativeAvgNorm = normSum/(samples+1);
disp(relativeAvgNorm);
