function [perturbation, perturbedVec] = PA(net, inputVec, labelVec, eta, cycles)
% PA - Perturbation Algorithm. PA perturbs an inputVec vector untill an NN can not correctly classify the inputVec
% Args:
%      net - a neural network which has methods forwardProp() and backProp().
%            forwardProp() must return a classification vector.
%            backProp() must return the last gradient dC/dX for X being an inputVec to the NN.
%            The last layer of the NN should not have softmax
%      inputVec - an inputVec vector for classification
%      labelVec - a true label for an inputVec vector
%      eta - a step size of each perturbation iteration
%      cycles - number of tries of perturbations
%
    perturbedVec = inputVec;

    if min(size(inputVec))~= 1
        error('Input is not a vector');
    end

    perturbation = 0;
    count = 0;

    [~,labelIndex] = max(labelVec);

    net.forwardProp(inputVec);      % update the NN's y^(l) neuron values

    while count < cycles
        dCdX = net.backProp(inputVec, labelVec, eta, false);
        perturbation = perturbation + eta * dCdX;           % compute perturbation vector
        perturbedVec = inputVec + perturbation;         % perturb the inputVec

        classifRes = ActivFunc.softmax(net.forwardProp(perturbedVec))';

        [prediction, predictionInd] = max(classifRes);

        % Return if fooled
        if predictionInd ~= labelIndex
            disp('fooled');
            return;
        end

        count = count + 1;
    end

    disp('Could not perturb the inputVec.')
    perturbation = zeros(size(inputVec));
    perturbedVec = zeros(size(inputVec));
end
