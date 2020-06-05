clear all; clc; close all;

%Design Params
lambda =  0.1;
GDparams.n_epochs = 40;
GDparams.n_batch = 100;
GDparams.eta = 0.001;

%Run settings (Use all zeros for "vanilla" run)
trainseveralmodels = 0;
usemoredata = 0;
XavierInit = 0;
GDparams.etascaling = 0;
GDparams.ordershuffle = 0;



%Load Data
if usemoredata
    [trainX1, trainY1hot1, trainY1] = LoadBatch("data_batch_1.mat");
    [trainX2, trainY1hot2, trainY2] = LoadBatch("data_batch_2.mat");
    [trainX3, trainY1hot3, trainY3] = LoadBatch("data_batch_3.mat");
    [trainX4, trainY1hot4, trainY4] = LoadBatch("data_batch_4.mat");
    [trainX5, trainY1hot5, trainY5] = LoadBatch("data_batch_5.mat");
    
    X = [trainX1 trainX2 trainX3 trainX4 trainX5];
    Y1hot = [trainY1hot1 trainY1hot2 trainY1hot3 trainY1hot4 trainY1hot5];
    Y = [trainY1 trainY2 trainY3 trainY4 trainY5];
    
    [trainX,trainY1hot,trainY, devX, devY1hot, devY] = traindevsplit(98, X, Y1hot, Y);
    [testX, testY1hot, testY] = LoadBatch("test_batch.mat");
else
    [trainX, trainY1hot, trainY] = LoadBatch("data_batch_1.mat");
    [devX, devY1hot, devY] = LoadBatch("data_batch_2.mat");
    [testX, testY1hot, testY] = LoadBatch("test_batch.mat");
end

%Normalize the data
mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);
devX = devX - repmat(mean_X, [1, size(devX, 2)]);
devX = devX ./ repmat(std_X, [1, size(devX, 2)]);
testX = testX - repmat(mean_X, [1, size(testX, 2)]);
testX = testX ./ repmat(std_X, [1, size(testX, 2)]);



% Exercise 2
% Initialize W and b
[d,N] = size(trainX);
[K,~] = size(trainY1hot);
if XavierInit
    W = sqrt(1/ N) .* randn(K,d) ;
else
    rng(400)
    W = sqrt(0.01) .* randn(K,d);
end
b = sqrt(0.01) .* randn(K,1);

% Train and evaluate model
[W_opt, b_opt] = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, W, b, lambda);
acctrain = ComputeAccuracy(trainX, trainY, W_opt, b_opt);
disp("Accuracy on the train data is: " + acctrain);

acctest = ComputeAccuracy(testX, testY, W_opt, b_opt);
disp("Accuracy on the test data is: " + acctest);

% Train more models
if trainseveralmodels
    W2 = sqrt(0.01) .* randn(K,d).*sqrt(1/N);
    b2 = sqrt(0.01) .* randn(K,1);
    W3 = sqrt(0.01) .* randn(K,d).*sqrt(1/N);
    b3 = sqrt(0.01) .* randn(K,1);
    [W_opt2, b_opt2] = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, W2, b2, lambda);
    [W_opt3, b_opt3] = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, W3, b3, lambda);
    accseveral = ComputeAccuracySeveralModels(trainX, trainY, W_opt, b_opt,W_opt2, b_opt2,W_opt3, b_opt3);
    disp("Accuracy for several models is: " + accseveral);
end

%Display trained models
for i=1:10
    im = reshape(W_opt(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure(2)
montage(s_im);


%% Test code

Jtest = ComputeCost(trainX(:, 1:100),trainY1hot(:, 1:100), W, b,lambda);
acctest = ComputeAccuracy(trainX(:, 1:100), trainY(:, 1:100), W, b);

Ptest = EvaluateClassifier(trainX(1:20, 1:10),W(:, 1:20),b);
[grad_b, grad_W] = ComputeGradients(trainX(1:20, 1:10), trainY1hot(:, 1:10),Ptest(:, 1:10), W(:, 1:20), lambda);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(1:20, 1:10), trainY1hot(:, 1:10), W(:, 1:20), b, lambda, 1e-6);

verification = norm(grad_b-ngrad_b,1)/max(1e-6,norm(grad_b,1)+norm(ngrad_b,1)); %verify that this is small


%% Exercise 7
function [Wstar, bstar] = MiniBatchGD(Xtrain, Ytrain, Xdev, Ydev, GDparams, W, b, lambda)
    [~,N] = size(Xtrain);
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    Wstar = W;
    bstar = b;
    loss = zeros(2,n_epochs);
    ind = linspace(1,N,N);
    for epoch = 1:n_epochs
        if GDparams.ordershuffle
            ind = randperm(N); %Randomize order
        end
        %disp(epoch +"/" + n_epochs)
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xtrain(:, ind(j_start:j_end));
            Ybatch = Ytrain(:, ind(j_start:j_end));
            
            P = EvaluateClassifier(Xbatch,Wstar,bstar);
            
            noise = 
            
            [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch,P, Wstar, lambda);
            
            Wstar = Wstar-eta*grad_W;
            bstar = bstar-eta*grad_b;
        end
        
        loss(:,epoch) = [ComputeCost(Xtrain, Ytrain, Wstar, bstar, lambda); ComputeCost(Xdev, Ydev, Wstar, bstar, lambda)];
        if GDparams.etascaling
            eta = eta*0.9;
        end
        disp("Training loss is: "+loss(1,epoch))
    end
    figure(1)
    plot(1:n_epochs,loss(1,:),1:n_epochs,loss(2,:))
    legend('Training set','Dev set')
    title('Loss over epochs')
    xlabel('Epoch')
    ylabel('Loss')
end

%% Exercise 1
% Load data from files
function [X, Y, y] = LoadBatch(filename)
    path = matlab.desktop.editor.getActiveFilename;
    [filepath,~,~] = fileparts(path);
    filepath = filepath + "/Datasets/cifar-10-batches-mat/";
    addpath(filepath);
    A = load(filename);
    y = A.labels'+1;
    Y = bsxfun(@eq, y(:), 1:max(y))';

    X = double(A.data')/255;
end

%% Exercise 3
% Softmax of model
function P = EvaluateClassifier(X,W,b)
    [~,N] = size(X);
    [K,~] = size(W);
    P = zeros(K,N);
    for i = 1:N
        s = W*X(:,i)+b;
        P(:,i) = exp(s)/sum(exp(s))';
    end
end

%% Exercise 4
% Calculate loss
function J = ComputeCost(X, Y, W, b, lambda)
    [~,N] = size(X);
    [K,~] = size(W);
    
    P = EvaluateClassifier(X,W,b);
    crossEntropy = 0;
    for i = 1:N
        crossEntropy = crossEntropy - log(Y(:,i)'*P(:,i));
    end
    regTerm = lambda*W(:)'*W(:);
    
    J = crossEntropy/N+regTerm;
end
%% Exercise 5
% Calculate accurace. Uses labels for Y.
function acc = ComputeAccuracy(X, Y, W, b)
    [~,N] = size(X);
    P = EvaluateClassifier(X,W,b);
    [~,I] = max(P);
    acc = numel(find(I==Y))/N;
end

%% Exercise 6
% Calculate the gradient for backpropogation
function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda)
    [~,N] = size(X);
    G = Y-P;
    grad_W = -G*X'/N+2*lambda*W;
    grad_b = -sum(G,2)/N;
end
%% Bonus points
% Split data in a randomized train and dev set
function [trainX,trainY1hot,trainY, devX, devY1hot, devY] = traindevsplit(percentagesplit,X,Y1hot,Y)
    [~,N] = size(X);
    testsize = floor(percentagesplit/100*N);
    ind = randperm(N);
    trainX = X(:, ind(1:testsize));
    trainY1hot = Y1hot(:, ind(1:testsize));
    trainY = Y(:, ind(1:testsize));
    devX = X(:, ind(testsize+1:end));
    devY1hot = Y1hot(:, ind(testsize+1:end));
    devY = Y(:, ind(testsize+1:end));
end

% Calculate accuracy of several models
function acc = ComputeAccuracySeveralModels(X, Y, W1, b1, W2 ,b2, W3, b3)
    [~,N] = size(X);
    P1 = EvaluateClassifier(X,W1,b1);
    P2 = EvaluateClassifier(X,W2,b2);
    P3 = EvaluateClassifier(X,W3,b3);
    [~,I1] = max(P1);
    [~,I2] = max(P2);
    [~,I3] = max(P3);
    
    I1 = bsxfun(@eq, I1(:), 1:max(I1))';
    I2 = bsxfun(@eq, I2(:), 1:max(I2))';
    I3 = bsxfun(@eq, I3(:), 1:max(I3))';
    
    I = I1+I2+I3;
    [~,I] = max(I);
    
    acc = numel(find(I==Y))/N;
end
%% Imported function
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end