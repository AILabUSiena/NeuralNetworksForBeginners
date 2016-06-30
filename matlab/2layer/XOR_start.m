
%  Quick start on XOR data

X = [0 0;0 1;1 0;1 1]'; % input matrix
Y = [0 1 1 0]; % target matrix

%   model variables
learningRate = 1e-3;
epochsOfTraining = 2.5e4;
hiddenUnits = 5;
inputSize = size(X,1);
outputSize = size(Y,1);

%   initialization
n = nnInit(hiddenUnits,inputSize,outputSize);

%   training
n = nnTrain(n,X,Y,epochsOfTraining,learningRate);

% plot performance
figure;

subplot(1,2,1);
plot(n.Loss);
xlabel('Epochs of Training','FontSize',14)
xlabel('MSE','FontSize',14)
title('Training Error','FontSize',16)

subplot(1,2,2);
plot(n.Accuracy);
xlabel('Epochs of Training','FontSize',14)
xlabel('AccuracyE','FontSize',14)
title('Training Classification Accuracy','FontSize',16)