classdef ResNetAntiSym < ResNetCustom
    properties
        K;
        G;
        igamma;
    end

    methods
        function obj = ResNetAntiSym(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, i_gamma, h, initScaler, i_testMode, activFunc, p, s, r, r1, r2)

            obj@ResNetCustom(i_numHiddenLayers, i_inputLayerSize, i_outputLayerSize, i_hiddenLayersSize, h, initScaler, i_testMode, activFunc, p, s, r, r1, r2);

            obj.igamma = i_gamma;
            obj.G = obj.igamma * eye(obj.hiddenLayersSize);
            obj.K{1} = 0;

            % Build intermediate W and b
            for i = 3:obj.numHiddenLayers + 2   % do not build W and b from last hidden layer to output layer
                obj.K{i} = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize, obj.hiddenLayersSize]);
                obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                obj.b{i} = obj.initScaler*normrnd(0,1,[obj.hiddenLayersSize,1]);
            end
        end


        function backresult = backProp(obj, i_vector, label_vector, eta, updateWeights)
            % Back propagation. Inputs: self, training vector, label vector, learning rate eta, updateWeights: true/false
            YN = obj.totalNumLayers;
            % Build softmax layer
            h_vec = ActivFunc.softmax(obj.Y{end});

            % Build dh/dY^(L) matrix, i.e deriv of softmax h w.r.t y^(L)
            dh = [];
            for i =1:obj.outputLayerSize;
                for j=1:obj.outputLayerSize;
                    if i==j
                        dh(i,j) = h_vec(i)*(1-h_vec(j));
                    else
                        dh(i,j) = -h_vec(i)*h_vec(j);
                    end
                end
            end

            % Calculate the last layer error gradient dC/dY^(L)
            obj.D{YN} = dh' * (-label_vector ./ h_vec');
            % Calculate the last layer weights gradient dY^(L)/dW^(L)
            obj.O{YN} = (obj.h*obj.D{YN}.*obj.df(obj.W{YN},obj.Y{YN-1},obj.b{YN}))*obj.Y{YN-1}';
            % Calculate YN-1 layer gradient
            obj.D{YN-1} = obj.WYN_lin'*obj.D{YN} + obj.W{YN}'*(obj.D{YN}.*obj.df(obj.W{YN},obj.Y{YN-1},obj.b{YN})*obj.h);
            % Calculate the last YN-1 layer weights gradient dY_(L-1)/dW_(L-1)
            obj.O{YN-1} = obj.h * obj.df(obj.W{YN-1}, obj.Y{YN-2}, obj.b{YN-1}) * obj.Y{YN-2}';

            % Calculate error gradient for L-2, L-3,..., 2 layers
            for i = YN-2:-1:2
                % Compute delta
                obj.D{i} = obj.D{i+1} + obj.W{i+1}'*( obj.D{i+1} .* (obj.h*obj.df(obj.W{i+1},obj.Y{i},obj.b{i+1})) );
                % Compute omega
                obj.O{i} = obj.h * obj.df(obj.W{i},obj.Y{i-1}, obj.b{i}) * obj.Y{i-1}';
            end

            % Compute gradient dC/dX
            obj.D{1} = obj.W2_lin' * obj.D{2} + obj.h*obj.W{2}' * (obj.D{2} .* obj.df(obj.W{2}, i_vector, obj.b{2}));
            backresult = obj.D{1};  % return dC/dX

            if updateWeights == true
                % Gradient step. Update weights and biases

                % First update dim reduction weights and biases
                % obj.W2_lin = obj.W2_lin - eta*obj.D{2} *i_vector' - eta*obj.r1*2*obj.W2_lin;
                % obj.b2_lin = obj.b2_lin - eta*obj.D{2};

                obj.WYN_lin = obj.WYN_lin - eta*obj.D{YN} * obj.Y{YN-1}';
                obj.bYN_lin = obj.bYN_lin - eta*obj.D{YN};

                % Update ReLu weights and biases for layer 2 and YN
                obj.W{2} = obj.W{2} - eta * (obj.h*obj.D{2}.* obj.df(obj.W{2}, i_vector, obj.b{2}))* i_vector' - eta*obj.r1*2*obj.W{2};
                obj.b{2} = obj.b{2} - eta* obj.h* obj.D{2} .* obj.df(obj.W{2}, i_vector,obj.b{2});

                obj.W{YN} = obj.W{YN} - eta * obj.O{YN} - eta*obj.r2*2*obj.W{YN};
                obj.b{YN} = obj.b{YN} - eta* obj.h* obj.D{YN} .* obj.df(obj.W{YN}, obj.Y{YN-1}, obj.b{YN});

                dRdb = 0; dRdK = 0;
                % Update intermediate layers
                for i = 3:YN-1

                    % Regularization
                    if obj.r ~= 0
                        if i==3
                            V = obj.W{i+1} - obj.W{i};
                            dRdb = -2*obj.r*(obj.b{i+1} - obj.b{i});
                        elseif i==YN-1
                            V = obj.W{i} - obj.W{i-1};
                            dRdb = -2*obj.r*(obj.b{i} - obj.b{i-1});
                        else
                            V = obj.W{i+1} - 2*obj.W{i} + obj.W{i-1};
                            dRdb = -2*obj.r*(obj.b{i+1} -2*obj.b{i} + obj.b{i-1});
                        end
                        dRdK = obj.r*(-V + V');
                    end

                    obj.K{i} = obj.K{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})') - eta*dRdK;
                    % obj.K{i} = obj.K{i} - eta * 0.5*(diag(obj.D{i})*obj.O{i} - (diag(obj.D{i})*obj.O{i})') - eta*obj.r*(obj.W{i} - obj.W{i}');
                    obj.W{i} = 0.5*(obj.K{i} - obj.K{i}' - obj.G);
                    obj.b{i} = obj.b{i} - eta* obj.h* obj.D{i} .* obj.df(obj.W{i}, obj.Y{i-1}, obj.b{i}) - eta*dRdb;
                end
            end
        end


    end
end
