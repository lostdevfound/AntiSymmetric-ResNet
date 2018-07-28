classdef ResNetAntiSymODE < ResNetCustom
    properties
        K;
        G;
        igamma;
    end

    methods
        function obj = ResNetAntiSymODE(i_numHiddenLayers, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode, activFunc, p, s, r)

            obj@ResNetCustom(i_numHiddenLayers, i_hiddenLayersSize, i_hiddenLayersSize, i_hiddenLayersSize, h, initScaler, i_testMode, activFunc, p, s, r);

            obj.igamma = i_gamma;
            obj.G = obj.igamma * eye(obj.hiddenLayersSize);
            obj.K{1} = 0;

            % Build intermediate W and b
            for i = 2:obj.numHiddenLayers + 1   % do not build W and b from last hidden layer to output layer
                obj.K{i} = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.hiddenLayersSize]);
                obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                obj.b{i} = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);
            end
        end


        function result = forwardProp(obj, i_vector)
            % Forward propagation
            YN = obj.totalNumLayers;
            obj.Y{1} = i_vector;

            for i=2:YN
                obj.Y{i} = obj.Y{i-1} + obj.h*obj.f(obj.W{i},obj.Y{i-1},obj.b{i});
            end

            result = obj.Y{end};

        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false
            obj.Y{1} = i_vector;
            YN = obj.totalNumLayers;

            % Build softmax layer
            h_vec = ActivFunc.softmax(obj.Y{end});

            % Build dh/dY^(L) matrix, i.e deriv of softmax h w.r.t y^(L)
            dh = [];
            for i =1:obj.outputLayerSize
                for j=1:obj.outputLayerSize
                    if i==j
                        dh(i,j) = h_vec(i)*(1-h_vec(j));
                    else
                        dh(i,j) = -h_vec(i)*h_vec(j);
                    end
                end
            end

            % Calculate the last layer error gradient dC/dY^(L) (CHECKED)
            obj.D{YN} = dh' * (-label_vector ./ h_vec');
            % Calculate the last layer weights gradient dY^(L)/dW^(L)
            obj.O{YN} = obj.h * obj.df(obj.W{YN},obj.Y{YN-1}, obj.b{YN}) * obj.Y{YN-1}';

            % Calculate error gradient for L-1, L-2,..., 2 layers
            for i = YN-1:-1:2
                % Compute delta (CHECKED)
                obj.D{i} = obj.D{i+1} + obj.W{i+1}'*( obj.D{i+1} .* (obj.h*obj.df(obj.W{i+1},obj.Y{i},obj.b{i+1})) );
                % Compute omega
                obj.O{i} = obj.h * obj.df(obj.W{i},obj.Y{i-1}, obj.b{i}) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX (CHECKED)
            obj.D{1} = obj.D{2} + obj.h*obj.W{2}' * (obj.D{2} .* obj.df(obj.W{2}, i_vector, obj.b{2}));
            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases
                V = 0; dRdK = 0; dRdb = 0;
                for i = 2:YN
                    % Gradient update
                    if i==2
                        V = obj.W{3} - obj.W{2};
                        dRdb = -2*obj.r*(obj.b{3} - obj.b{2});
                    elseif i==YN
                        V = obj.W{YN} - obj.W{YN-1};
                        dRdb = -2*obj.r*(obj.b{YN} - obj.b{YN-1});
                    else
                        V = obj.W{i+1} - 2*obj.W{i} + obj.W{i-1};
                        dRdb = -2*obj.r*(obj.b{i+1} -2*obj.b{i} + obj.b{i-1});
                    end
                    dRdK = obj.r*(-V + V');
                    % dRdW = -2*obj.r*V;
                    % obj.K{i} = obj.K{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})') - eta*dRdK;
                    obj.K{i} = obj.K{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})') - eta*obj.r*(obj.W{i} - obj.W{i}');
                    obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                    % obj.b{i} = obj.b{i} - eta* obj.h* obj.D{i} .* obj.df(obj.W{i}, obj.Y{i-1}, obj.b{i}) -eta*dRdb;
                    obj.b{i} = obj.b{i} - eta* obj.h* obj.D{i} .* obj.df(obj.W{i}, obj.Y{i-1}, obj.b{i});
                end
            end
        end

        
    end
end
