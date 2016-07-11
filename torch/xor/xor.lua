require "nn"
require "gnuplot"

---- Neural Network Creation ----
mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 20; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

---- Loss Function ----
criterion = nn.MSECriterion()

---- Training the Network ----
for i = 1,35000 do
  -- random sample
  local input= torch.randn(2);     -- normally distributed example in 2d
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then  -- calculate label for XOR function
    output[1] = -1
  else
    output[1] = 1
  end

  -- feed it to the neural network and the criterion
  criterion:forward(mlp:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, criterion:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.05)
end

---- Test the Network ----
x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

----  Better Test the Network ----

local class0, class1 = {x= {}, y={}},{x= {}, y={}}
for i = 1,9000 do
  -- random sample
  local x= torch.randn(2);     -- normally distributed example in 2d
  if torch.pow(x[1],2) < 1 and  torch.pow(x[2],2) < 1 then
    if mlp:forward(x)[1] > 0 then  -- calculate label for XOR function
      table.insert(class0.x,x[1])
      table.insert(class0.y,x[2])
      print(x)
    else
      table.insert(class1.x,x[1])
      table.insert(class1.y,x[2])
    end
  end
end

--gnuplot.epsfigure('XOR.eps')
gnuplot.plot({'predicted as true', torch.Tensor(class0.x),torch.Tensor(class0.y),'+'},{'predicted as false', torch.Tensor(class1.x),torch.Tensor(class1.y),'+'})
gnuplot.title('Xor');
--    gnuplot.plotflush()
gnuplot.figure()



