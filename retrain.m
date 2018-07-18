% get the Network
% Reinit weight entries above the threshold with random values
% Call training but with modified weight update

function net = retrain(inet, trainData, trainLabel, cycles, eta, retrainCycles)
    for k=1:retrainCycles
        numWeightsAboveThresh = 0;
        maxConcavPerW = [];

        for i=2:inet.totalNumLayers
            % Get concavity matrix
            concavW = inet.computeConcavityW(i);
            maxConcavPerW(i) = max(max(concavW));
            threshold = maxConcavPerW(i) * 0.4;
            % Init random matrix
            R = inet.initScaler*normrnd(0,1,size(concavW));
            % Init mask
            M = ones(size(concavW));
            % Find weights with concavity above threshold
            indVec = find(concavW > threshold);
            numWeightsAboveThresh = numWeightsAboveThresh + numel(indVec);

            for j=1:numel(indVec)
                inet.M{i}(indVec(j)) = 0;
                % M(indVec(j)) = 0;
            end

            % M_reverse = ones(size(concavW))- M;
            % Randomize weights that are above the threshold and update masks
            inet.W{i} = inet.W{i} .* inet.M{i};
            % inet.W{i} = inet.W{i} .* M + R .* M_reverse;
            % inet.M{i} = M_reverse;

        end

        disp('num weights above tresh:');
        numWeightsAboveThresh

        % inet.train(trainData, trainLabel, cycles, eta);
        net = inet;
    end
end
