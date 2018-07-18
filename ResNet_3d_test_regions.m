clc;
clear all;
close all
% Load training data
load('resources/data3d/data3d.mat');
load('resources/data3d/label3d.mat');
trainLabelSet = labelSet(:,1:25000);
trainDataSet = dataSet(:,1:25000);

validLabelSet = labelSet(:,25001:30000);
validDataSet = dataSet(:,25001:30000);

% Setup NN's params
igamma = 0.0001;       % default 0.1
trainCycles = 400000;       % default 400000
eta = 0.01;           % good default 0.0005 or 0.003
initScaler = 1;      % default 0.01
neurons = 3;
layers = 20;
h = 0.5;        % default 0.7


% Set to true if need to retrain
first_time_launch = false;
doPerturbation = false;

% Training part.
if first_time_launch == true
    % Init NN and train it. Params i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode
    net = ResNetAntiSymODE(layers, 3, 3, neurons, igamma, h, initScaler, false);
    disp('training...');
    net.train(trainDataSet, trainLabelSet, trainCycles, eta);
    disp('training complete.');

    % Save trained net
    netStr = {'ODE_net_l', num2str(layers), '_h', num2str(h), '_ig', num2str(igamma), '_n', num2str(neurons), '.mat','resources/'};
    str = strcat(netStr{10},netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9});
    net.name = strcat(netStr{1},netStr{2},netStr{3},netStr{4},netStr{5},netStr{6},netStr{7},netStr{8}, netStr{9});
    save(str,'net');
else
     load('resources/ODE_net_l20_h0.5_ig0.0001_n3.mat')
    %load('resources/softmax_net_l10_h0.2_n3.mat')
end

%% generate fine grid in the cube [-1,1]^3
GridSize = 200;
grid_1D = linspace(-1,1,GridSize);
[X_grid,Y_grid,Z_grid] = meshgrid(grid_1D,grid_1D,grid_1D);
x_grid = X_grid(:);
y_grid = Y_grid(:);
z_grid = Z_grid(:);

[X_plane,Y_plane] = meshgrid(grid_1D,0.5*grid_1D);
Z_plane = 2*Y_plane;
x_plane = X_plane(:);
y_plane = Y_plane(:);
z_plane = Z_plane(:);


ClassifResults = [];
for i_point = 1: length(x_plane)
    InputVector = [x_plane(i_point); y_plane(i_point); z_plane(i_point)];
    OutputVector = softmax(net.forwardProp(InputVector));
    ClassifResults = [ClassifResults, OutputVector];
    
    if OutputVector(1) > OutputVector(2)
        Values_vector(i_point) = 1;
        %scatter3(InputVector(1),InputVector(2),InputVector(3),100,[1 0 0],'*');%,'color','red');
    else
        Values_vector(i_point) = 0;
        %scatter3(InputVector(1),InputVector(2),InputVector(3),100,[0 0 1],'*');%,'color','blue');
    end
    %hold on
end
%hold off
Values = reshape(Values_vector,GridSize,GridSize);

surf(X_plane,Y_plane,Z_plane,Values)




