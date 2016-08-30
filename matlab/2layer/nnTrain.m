function nn = nnTrain(nn,X,Y,maxEpochs,eta)
% nnTrain : perform an on-line neural network training on the structure
%           'nn' on the input X with target 'Y'
%
%     nn = nnTrain(nn,X,Y,maxEpochs)
%
%     nn : structure containing variables to implement a simple Neural
%          Network (see 'nnInit','nnEval' for variables explanation)
%     X : (input_size)-by-(number_of_samples) data matrix
%     Y : (output_size)-by-(number_of_samples) target matrix for X
%     maxEpochs : max number of training epochs
%     eta : learning rate

if size(Y,1)>1
    labels = vec2ind(Y); % labels for multi-class accuracy
else
    cT = (max(Y)+min(Y))/2; % classes threshold for 2-class accuracy
    labels = Y>cT;
end

i = 1;

while i <= maxEpochs
    
    for j = 1 : size(X,2)
        
        %   forward propagation
        nn = nnEval(nn,X(:,j));
        
        %   Square Error derivative evaluation
        delta = nn.o - Y(:,j);
        
        %   gradients computing
        % dE/dwO = (dE/do)*(do/dwO)
        nn.D_wO = delta * nn.zH'; % output layer weights
        % dE/dbO = (dE/do)*(do/dbO)  (do/dbO = 1)
        nn.D_bO = delta; % output layer bias
        % back-propagation
        % dE/dbH = (dE/do)*(do/dzH)*(dzH/daH)*(daH/dbH) , (daH/dbH = 1)
        nn.D_bH = (nn.wO'*delta).*(nn.aH>0); % hidden layer bias
        % dE/dwH = (dE/do)*(do/dzH)*(dzH/daH)*(daH/dwH)
        nn.D_wH = nn.D_bH * X(:,j)' ; % hidden layer weights 
        
        %   updating
        nn.wO = nn.wO - eta*nn.D_wO;
        nn.bO = nn.bO - eta*nn.D_bO;
        nn.wH = nn.wH - eta*nn.D_wH;
        nn.bH = nn.bH - eta*nn.D_bH;
         
    end
    
    %   error evaluating
    nn = nnEval(nn,X);
    MSE = 0.5*mean(mean((nn.o-Y).^2,1),2);
    nn.Loss = [nn.Loss,MSE];
    %   classification accuracy
    if size(Y,1)>1
        Accuracy = mean(vec2ind(nn.o) == labels);
    else
        Accuracy =  mean((nn.o>cT)==labels);
    end
    nn.Accuracy = [nn.Accuracy,Accuracy];
    
    fprintf('Epoch of training: %i/%i - Error: %f \n',i,maxEpochs,nn.Loss(end));
    
    if (nn.Loss(end)<1e-4)&&(nn.Accuracy(end)>0.99) % stopping criterion
        break;
    end
    
    i = i + 1;
    
end



end
