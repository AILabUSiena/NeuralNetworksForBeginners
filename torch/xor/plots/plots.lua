function plotSepSurface( mlp, n )
    -- estimate the separation surface
    local eps = 2e-3
    local  n = n or 1000000 
    local x = torch.rand(n,2):add(-0.5) -- huge number of 2D samples in [-0.5,0.5]
    local mlpOutput = mlp:forward(x) -- predict the value of each sample
    local mask = torch.le(torch.abs(mlpOutput),mlpOutput:clone():fill(eps)) -- a mask of x to plot only samples predicted less or equal than eps

    if torch.sum(mask) > 0 then
        gnuplot.epsfigure('Separation surface.eps')
        local x1 = x:narrow(2,1,1)
        local x2 = x:narrow(2,2,1)
        gnuplot.plot(x1[mask], x2[mask], '+')
        gnuplot.title('Separation surface')
        gnuplot.grid(true)
        gnuplot.plotflush()
        gnuplot.figure()
    end
end

function plotPredictions(mlp, n)
    -- given n samples randomly picked, plots positive and negative
    -- predictions with different colors
    local n = n or 3000
    local x = torch.rand(n,2):add(-0.5)
    local mlpOutput = mlp:forward(x)
    local truePredMask = torch.gt(mlpOutput, mlpOutput:clone():fill(0))
    local falsePredMask = truePredMask:clone():add(-1):mul(-1)
    local x1 = x:narrow(2,1,1)
    local x2 = x:narrow(2,2,1)

print(falsePredMask)
print(mlpOutput)
    if torch.sum(truePredMask) > 0 and  torch.sum(falsePredMask) > 0 then
        local truePlot = {'predicted as true', x1[truePredMask], x2[truePredMask],'+'}
        local falsePlot = {'predicted as false', x1[falsePredMask], x2[falsePredMask], '+'}
        gnuplot.epsfigure('XOR.eps')
        gnuplot.plot(truePlot,falsePlot)
        gnuplot.title('Xor')
        gnuplot.grid(true)
        gnuplot.plotflush()
        gnuplot.figure()
    end
end

function plotLoss( loss, weightDecay )
    -- compute the loss function and the weight decay, over the number of epochs
    local nepochs = loss:size(1) > weightDecay:size(1) and loss:size(1) or weightDecay:size(1)
    gnuplot.epsfigure('XORloss.eps')
    gnuplot.plot({'loss function', torch.range(1,nepochs),loss},{'weight decay', torch.range(1,nepochs), weightDecay})
    gnuplot.title('Loss')
    gnuplot.grid(true)
    gnuplot.plotflush()
    gnuplot.figure()
end