# Setting up the XOR problem

In this repository we show our simple implementation of a 2-layer ANN, just to give you some tips if you need (or would like) to design your own model. The code we present is basic and can be easily improved, but we try to keep it simple just to understand fundamental steps. In back-propagation we avoid for loops exploiting the Matlab efficiency with matrix operations. This is a key points and can substantially affect the running time for large data.

## Data definition
The classification task on the boolean function XOR is a common not-linearly separabile instructional Machine Learning problem. Input data can be defined as a matrix:
```matlab
   >> X = [0, 0, 1, 1; 0, 1, 0, 1]
   
   X =

       0     0     1     1
       0     1     0     1
   ```
To which we can assign the target:
```matlab
   >> Y = [0, 1, 1, 0]
   
   Y =

       0     1     1     0
   ```


## Network initialization

In our case for example, we save the variables of the network in a structure (named `nn`) for which we define three functions. [nnInit.m](https://github.com/alered87/First-Day-at-AI-lab/blob/master/matlab/2layer/nnInit.m) randomly initializes the weights and bias of the hidden (wH, bH) and the output (wO, bO) layers. Each input sample is assumed in column vector form, then the weights are organized in matrices with as many rows as the number of units in the layer (the desired number of hidden units in the hidden layer and the output size in output layer) and as many columns as the dimension of the input (the data dimension for the hidden layer and the number of hidden units in the output one). We can create a network to compute the XOR data passing the number of hidden units (10 in this case), the input and the output dimension to:
```matlab
 nn = nnInit(10,2,1)
 ```

The prediction of the network on data can be evaluated by  can by [nnEval.m](https://github.com/alered87/First-Day-at-AI-lab/blob/master/matlab/2layer/nnEval.m). This can be set by simple matrices multiplications, since the built-in Matlab functions which are used to compute the transfer function (i.e. _logsig_, _tanh_, etc) are vectorized (for example we implement the _ReLu_ activation by `max(A,0)`). The function [nnTrain.m](https://github.com/alered87/First-Day-at-AI-lab/blob/master/matlab/2layer/nnTrain.m) perform the training of the network with respect to the given input data and targets. We set up the training phase in an on-line setting via a for loops, but its straight-forward to implement a batch training by matrices multiplications. To start the training we have to provide as argument the structure containing variable (`nn``)), the defined input data `X` and its targets `Y`, the desired number of epochs of training and the learning rate, obtaing as output the trained structure:


```matlab
 nn = nnTrain(nn,X,Y,1000,0.001)
 ```
We can visualize performance on data by ...

<img src="perf.pdf" alt="performance plot"/><br/>
