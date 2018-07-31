classdef Regularization
    properties
        typeReg;
    end

    methods (Static)

        function dRdW = L2(W)
            dRdW = 2*W;
        end


        function dRdW = L1(W)
            dRdW = sign(W);
        end


        function dRdW = difInter(W_prev, W, W_next)
            columnSum = @Regularization.columnSum;

            [n,~] = size(W);
            c = (1.0/n);
            dRdW = -c*(W_prev - c*columnSum(W)) + W - c*columnSum(W_next);
        end


        function dRdW = difStart(W, W_next)
            columnSum = @Regularization.columnSum;
            [n,~] = size(W);
            c = (1.0/n);
            dRdW = W - c*columnSum(W_next);
        end


        function dRdW = difEnd(W_prev, W_L)
            columnSum = @Regularization.columnSum;
            [n,~] = size(W_L);
            c = (1.0/n);
            dRdW = -c*(W_prev -c*columnSum(W_L));
        end


        function columnSumMatrix = columnSum(W)
            [numRow, numCol] = size(W);

            for i=1:numCol
                columnSumVector(i) = sum(W(:,i));
            end

            columnSumMatrix = repmat(columnSumVector, numRow, 1);
        end
    end
end
