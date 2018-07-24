classdef ActivFunc < handle
    properties
        activ;
        testmode;
        p;
        s;
    end

    methods

        function obj = ActivFunc(activ, testmode,p ,s);
            obj.activ = activ;
            obj.testmode = testmode;
            obj.p = p;
            obj.s = s;
        end

        function y = activf(obj, W, x, b)
            % This function picks desired activation function based on th class property "activ"
            if strcmp(obj.activ, 'relu')
                y = ActivFunc.relu(W, x, b, obj.testmode);

            elseif strcmp(obj.activ,'sigmoid')
                y = ActivFunc.v_sigm(W, x, b);

            elseif strcmp(obj.activ, 'powerlog')
                y = ActivFunc.powerlog(W, x, b, obj.p, obj.s);

            elseif strcmp(obj.activ, 'sqrf')
                y = ActivFunc.sqrf(W, x, b);
            end
        end


        function d = activfD(obj, W, x, b)
            % This function picks desired activation function based on the string parameter "activ"
            if strcmp(obj.activ, 'relu')
                d = ActivFunc.reluD(W, x, b, obj.testmode);

            elseif strcmp(obj.activ,'sigmoid')
                d = ActivFunc.v_sigmD(W, x, b);

            elseif strcmp(obj.activ, 'powerlog')
                d = ActivFunc.powerlogD(W, x, b, obj.p, obj.s);

            elseif strcmp(obj.activ, 'sqrf')
                d = ActivFunc.sqrfD(W, x, b);
            end
        end


        function dd = activfDD(obj, W, x, b)
            if strcmp(obj.activ, 'sqrf')
                dd = ActivFunc.sqrfDD(W,x,b);
            end
        end
    end


    methods (Static)

        function y = powerlog(W, x, b, p, s)
            % vector log(z)^p activation function
            z = W*x+b;
            y = (z >= 0).*log(z/s + 1).^p;
        end

        function d = powerlogD(W, x, b, p, s)
            % vector derivative of powerlogD
            z = W*x+b;
            d = p*(z >= 0) .* log(z/s +1).^(p-1) ./ ((z/s +1)*s);
        end

        function y = relu(W, x, b, testmode)
        % vector ReLu activation fucntion
            [~,n] = size(W);
            z = W*x+b;

            if testmode == true
                y = z;    % this is for testing without max() operator
            else
                y = max(0, z);
            end
        end


        function d = reluD(W, x, b, testmode)
            % vector ReluD derivative of the ReLu function
            leak = 0;

            if testmode == true
                d = 1;    % this is for testing without max() operator
            else
                d = W*x + b >= 0;
                d(d==0) = leak;
            end
        end


        function y = v_sigm(W, x, b)
            % vector sigmoid activation function.
            y = 1./(1+exp(-(W*x+b)));
        end


        function d = v_sigmD(W, x, b)
            % vector sigmD is a derivative of the sigmoid function
            sigm = ActivFunc.v_sigm(W,x,b);
            d = sigm .* (1 - sigm);
        end

        function y = sigm(z)
            % scalar sigmoid activation function.
            y = 1./(1+exp(-z));
        end


        function d = sigmD(x)
            % scalar sigmD is a derivative of the sigmoid function
            sigm = ActivFunc.sigm(x);
            d = sigm * (1 - sigm);
        end


        function y = sqrf(W, x, b)
            z = W*x + b;
            y = (z > 0).*z.^(1/2.0);
        end


        function d = sqrfD(W, x, b)
            z = W*x + b;
            d = 1/2.0*(z > 0).*z.^(-1/2.0);
        end

        function d = sqrfDD(W, x, b)
            z = W*x + b;
            d = -1/4.0*(z > 0).*z.^(-3/2.0);
        end


        function resSoft = softmax(y_args)
            % This function computes softmax

            if min(size(y_args)) ~= 1
                error('input is not a vector in softmax')
            end

            % Vectorized version
            C = max(y_args);
            resSoft = exp(y_args - C) / (sum(exp(y_args -C)));
            resSoft = resSoft';

            % inputSize = length(y_args);
            % C = max(y_args);
            % y_argsSum = 0;
            % for i = 1:inputSize
            %     y_argsSum = y_argsSum + exp(y_args(i) - C);
            % end
            % for i = 1:inputSize
            %     resSoft(i) = exp(y_args(i) - C) / y_argsSum;
            % end
        end
    end
end
