
%  Quick start on XOR data

X = [0 0;0 1;1 0;1 1]'; % input matrix
Y = [0 1 1 0]; % target matrix
% uncomment to train on fuzzy distribution
% X = rand( 2, 100 );
% Y = ( X(1,:)-.5 ).*( X(2,:)-.5 ) < 0 ;

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



% % % % % plot Loss of network in time

figure;

% plot Loss of network in time
subplot(1,2,1);
plot(n.Loss);
xlabel('Epochs of Training','FontSize',14)
ylabel('MSE','FontSize',14)
title('Training Error','FontSize',16)


% plot separation surface
subplot(1,2,2);

bound = [-.5,2;-1,1.5]; % axis bound
step = .3; % number of evaluation point

% generating space grid
[xp1,xp2] = meshgrid(bound(1,1):step:bound(1,2),bound(2,1):step:bound(2,2));

% evaluation on space grid
f = zeros(size(xp1));
for i=1:size(xp1,1)
    for j=1:size(xp1,2)
        n = nnEval(n,[xp1(i,j);xp2(i,j)]);
        f(i,j) = n.o;
    end
end

pcolor(xp1,xp2,f); % plot of evaluation color
shading interp; % removing gridding from plot
colormap(jet); % setting colormap
hold on;
contour(xp1,xp2,f,[.5,.5],'LineWidth',2,'Color','k'); % drawing separation curve
% drawing data points 
X = [0 0;0 1;1 0;1 1]'; % input matrix
Y = [0 1 1 0]; % target matrix
scatter(X(1,[1,4]),X(2,[1,4]),200,'o','filled','MarkerEdgeColor','k','MarkerFaceColor','w');
scatter(X(1,[2,3]),X(2,[2,3]),200,'d','filled','MarkerEdgeColor','k','MarkerFaceColor','w');
% labeling data points
c = {'X_1','X_2','X_3','X_4'};
dx = [-.15, -.15, .1, .1];
dy = [-.1, .1, -.1, .1];
text(X(1,:)+dx, X(2,:)+dy, c, 'FontSize',14);
colorbar;

% plot labels
xlabel('X_1','FontSize',14)
ylabel('X_2','FontSize',14)

title('Separation Surfaces','FontSize',16);
h = legend({'Prediction','Classes Bound','Class 0','Class 1'},'Location','SouthEast');
set(h,'FontSize',14);
