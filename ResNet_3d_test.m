clc;
clear all;
% Load training data
load('resources/data3d/data3d.mat');
load('resources/data3d/label3d.mat');
trainLabelSet = labelSet(:,1:10000);
trainDataSet = dataSet(:,1:10000);

validLabelSet = labelSet(:,10001:30000);
validDataSet = dataSet(:,10001:30000);

% Setup NN's params
igamma = 0.0001;       % default 0.1
trainCycles = 400000;       % default 400000
eta = 0.01;           % good default 0.0005 or 0.003
initScaler = 1;      % default 0.01
neurons = 3;
layers = 20;
h = 0.5;        % default 0.7
activFunc = 'relu';
regular = 0;
p = 1;
s = 1;


% Set to true if need to retrain
first_time_launch = true;
doPerturbation = true;

%                                           %
%              Training part                %
%                                           %

if first_time_launch == true
    % Init NN and train it. Params i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode
    net = ResNetAntiSymODE(layers, 3, 3, neurons, igamma, h, initScaler, false, activFunc, regular, p, s);
    disp('training...');
    net.train(trainDataSet, trainLabelSet, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'resources/', 'ODE_', activFunc, '_net_l', num2str(layers), '_h', num2str(h), '_n', num2str(neurons), '_p', num2str(p), '_s', num2str(s), '_r', num2str(regular),'_gamma',num2str(igamma), '.mat'};
    [~,numNames] = size(netStr);

    str = '';
    for i=1:numNames
        str = strcat(str,netStr(i));
    end

    save(str{1},'net');
else

    load('resources/ODE_net_l20_h0.5_ig0.0001_n3.mat')
    % load('resources/softmax_net_l10_h0.2_n3.mat')
end

%                                           %
%              Perturbation part            %
%                                           %

normSum = 0;
samples = 50;
offset= 4;
label = 0;
breakCount = 20000;

foolCount = 0;
onenormVecNum = 0;

for k = offset:offset + samples
    % Pick image then forwardProp image and print result in the console.
    index = k;     % Pick some image by its index (digit 3 is index 33)
    testVec =  validDataSet(:,index);
    [~,labelIndex] = max(validLabelSet(:,index));

    % Skip vectors that lie outside the sphere
    if labelIndex == 2
        disp('skipping');
        continue;
    else
        onenormVecNum = onenormVecNum + 1;
    end

    perturbedVec = testVec;
    classifRes = ones(3,1);

    % Perturbation generation
    count = 0;

    while doPerturbation == true
        net.forwardProp(perturbedVec);
        perturbedVec = net.adversBackProp(perturbedVec,validLabelSet(:,index), 0.01);
        classifRes = ActivFunc.softmax(net.forwardProp(perturbedVec));

        [prediction,predictionInd] = max(classifRes);

        % Break if predictionInd is oncorrect
        if predictionInd ~= labelIndex
            break;

        elseif count > breakCount
            perturbedVec = testVec;
            samples = samples -1;
            disp('breaking...........');
            disp(k)
            break;
        end

        count = count + 1;
    end

    % Count the nunmber of vecs that have norm < 1 and fooled the NN
    if  count < breakCount && (predictionInd ~= labelIndex) && norm(perturbedVec) < 1 && norm(testVec) < 1
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
        legendStrOrig = {'original', ', vec norm:', num2str(norm(testVec))};
        legendStrPerturbed = {'perturbed ', ', vec norm:', num2str(norm(perturbedVec))};
        legendStrOrig = strcat(legendStrOrig{1:end});
        legendStrPerturbed = strcat(legendStrPerturbed{1:end});
        legScatter = legend({legendStrOrig, legendStrPerturbed}, 'FontSize', 12);
        leg_pos = get(legScatter,'position');
        set(legScatter,'position',[leg_pos(1),leg_pos(2),leg_pos(3)*1.2,leg_pos(4)*2]) ;
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
% Didsplay picked image
disp('### Overall report ###');
disp('Vectors sampled:');
disp(onenormVecNum);
fooledPercent = foolCount/onenormVecNum*100;
disp('Percent fooled:');
disp(fooledPercent);
disp('relativeAvgNorm:')
relativeAvgNorm = normSum/(samples+1);
disp(relativeAvgNorm);

% X(1) = testVec(1);
% X(2) = perturbedVec(1);
% Y(1) = testVec(2);
% Y(2) = perturbedVec(2);
% Z(1) = testVec(3);
% Z(2) = perturbedVec(3);
%
% figure
% [xs,ys,zs] = sphere;
% h = surfl(xs, ys, zs);
% colormap([0 0 0])
% set(h, 'FaceAlpha', 0.1);
% shading interp
% hold on
% scatter3(X,Y,Z, [100 100], [0 0 1; 1 0 0], 'o')
% % set(gca,'XLim',[-2 2],'YLim',[-2 2],'ZLim',[-2 2])
% set(gca, 'Projection','perspective')
