
load('resources/ODE_relu_net_l20_h0.5_n3_p1_s1_r0_gamma0.0001.mat')
%load('resources/softmax_net_l10_h0.2_n3.mat')

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
