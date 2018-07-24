clc;
clear all;
% Load training data
load('resources/data3d/data3d.mat');
load('resources/data3d/label3d.mat');
trainLabelSet = labelSet(:,1:10000);
trainDataSet = dataSet(:,1:10000);

validDataSet = dataSet(:,10001:30000);
validLabelSet = labelSet(:,10001:30000);

load('resources/ODE_relu_net_l15_h0.1_n3_p1_s1_r0_gamma0.0001.mat');

index = 138;
testVec1 = validDataSet(:,index);
labelVec1 = validLabelSet(:,index);

testVec2 = validDataSet(:,index-1);
labelVec2 = validLabelSet(:,index-1);

eta = 0.01; cycles = 10000;
[perturbation, perturbedVec] = PA(net, testVec1, labelVec1, eta, cycles);

classifOriginal1 = net.forwardProp(testVec1);
correctPropagation1 = net.getArrayY();

classifOriginal2 = net.forwardProp(testVec2);
correctPropagation2 = net.getArrayY();

classifPerturbed = net.forwardProp(perturbedVec);
fooledPropagation = net.getArrayY();

results = [classifOriginal1, classifOriginal2, classifPerturbed];
disp(results);

for i=1:net.totalNumLayers
    correctX1(i) = correctPropagation1{i}(1);
    correctY1(i) = correctPropagation1{i}(2);
    correctZ1(i) = correctPropagation1{i}(3);
    correctX2(i) = correctPropagation2{i}(1);
    correctY2(i) = correctPropagation2{i}(2);
    correctZ2(i) = correctPropagation2{i}(3);

    fooledX(i) = fooledPropagation{i}(1);
    fooledY(i) = fooledPropagation{i}(2);
    fooledZ(i) = fooledPropagation{i}(3);
end

X(1) = testVec1(1);
X(2) = perturbedVec(1);
X(3) = testVec2(1);
Y(1) = testVec1(2);
Y(2) = perturbedVec(2);
Y(3) = testVec2(2);
Z(1) = testVec1(3);
Z(2) = perturbedVec(3);
Z(3) = testVec2(3);
disp('index:');disp(index);

figure
[xs,ys,zs] = sphere;
surfPlot = surfl(xs, ys, zs);
colormap([0 0 0]);
set(surfPlot, 'FaceAlpha', 0.1);
shading interp;
hold on;
scatter3(correctX1,correctY1,correctZ1, 'MarkerEdgeColor','k','MarkerFaceColor',[0 0 1]);
hold on;
scatter3(correctX2,correctY2,correctZ2,'MarkerEdgeColor','k','MarkerFaceColor',[0 1 0]);
hold on;
scatter3(fooledX,fooledY,fooledZ, '*');
hold on;
% scatter3(X,Y,Z, [100 100], [0 0 1; 1 0 0], '^');
scatter3(X,Y,Z, [100 100 100], [0 0 1; 1 0 0; 0 0.6 0.3], '^');
legend({'true region','original'})
