clear all;
clc;
close all;
%load resources/Custom_tan_h_net_l2_h0.1_n2_p1_s1_r0_gamma1e-10.mat
load('ODE_tan_h_net_l15_h0.3_n3_p1_s1_r0.0001_gamma0.0001.mat')

grid_res = 101;
x_grid = linspace(-1,1,grid_res);
[X_grid,Y_grid] = meshgrid(x_grid,x_grid);
xx_grid = X_grid(:);
yy_grid = Y_grid(:);

for i_point = 1:length(xx_grid)
    inputVector = [xx_grid(i_point); yy_grid(i_point)];
    outputVector = softmax(net.forwardProp(inputVector));
    if outputVector(1) > outputVector(2)
        values_vector(i_point) = 1;
    else
        values_vector(i_point) = 0;
    end
end

Values = reshape(values_vector,grid_res,grid_res);

figure(1)
scatter(xx_grid',yy_grid',ones(1,length(xx_grid)),values_vector);


%hold on
%scatter(trainDataSet(1,:),trainDataSet(2,:),ones(1,NTrain),trainLabelSet(1,:))

