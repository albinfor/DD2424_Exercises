%% Initialize data with all variables
clear all; clc; close all;

%Design Params
lambda =  1.283e-4;
GDparams.nCycles = 3;
GDparams.n_batch = 100;
GDparams.etamin = 1e-5;
GDparams.etamax = 1e-1;
sizeHiddenLayer = 50;

%Run settings (Use all zeros for "vanilla" run)
usemoredata = 1;
GDparams.ordershuffle = 1;

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

mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);
trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);
devX = devX - repmat(mean_X, [1, size(devX, 2)]);
devX = devX ./ repmat(std_X, [1, size(devX, 2)]);
testX = testX - repmat(mean_X, [1, size(testX, 2)]);
testX = testX ./ repmat(std_X, [1, size(testX, 2)]);



% Initialize W and b
[d,N] = size(trainX);
[K,~] = size(trainY1hot);
M = sizeHiddenLayer;

%Calculates n
GDparams.n_s = 2*floor(N/GDparams.n_batch);
GDparams.n_epochs = ceil(2*GDparams.n_s*GDparams.nCycles*GDparams.n_batch/N);

[W1, b1] = initParams(0, 1/sqrt(d), M, d);
[W2, b2] = initParams(0, 1/sqrt(M), K, M);
Theta = {W1,b1,M,d,'ReLU'};
Theta(2,:) = {W2,b2,K,M,'softmax'};

%% Look for good Lambda

l_min = -5;
l_max = -3;
n_samples = 20;
accuracies = [];
Lambdas = [];
for j = 1:n_samples
    l = l_min + (l_max - l_min)*rand(1, 1);
    Lambda = 10^l;
    disp("Training run: "+j+", Lambda = "+Lambda);
    Thetaopt = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, Theta, Lambda);
    accdev = ComputeAccuracy(devX, devY, Thetaopt);
    accuracies = [accuracies; accdev];
    Lambdas = [Lambdas; Lambda];
end
figure(3)
scatter(Lambdas,accuracies)
set(gca,'xscale','log')
xlabel('Lambda')
ylabel('Test accuracy')
title('Dev accuracy vs regularization constant')
grid on

%% Run a normal training session
% Train and evaluate model
Thetaopt = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, Theta, lambda);
acctrain = ComputeAccuracy(trainX, trainY, Thetaopt);
disp("Accuracy on the train data is: " + acctrain);

acctest = ComputeAccuracy(testX, testY, Thetaopt);
disp("Accuracy on the test data is: " + acctest);

%Display trained models
Layers = size(Theta,1);
W = Thetaopt{1,1};
for i=1:M
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
    
figure(2)
montage(s_im);


%% Test code
ThetaTest = Theta;
W = ThetaTest{1,1};
ThetaTest{1,1} = W(:, 1:20);

Ptest = EvaluateClassifier(trainX(1:20, 1:10),ThetaTest);
[grad_b, grad_W] = ComputeGradients(trainX(1:20, 1:10), trainY1hot(:, 1:10),Ptest, ThetaTest, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNum(trainX(1:20, 1:10), trainY1hot(:, 1:10), ThetaTest, lambda, 1e-6);

grad_b2 = grad_b{2};
grad_W2 = grad_W{2};
ngrad_b2 = ngrad_b{2};
ngrad_W2 = ngrad_W{2};
verificationb2 = norm(grad_b2-ngrad_b2,1)/max(1e-6,norm(grad_b2,1)+norm(ngrad_b2,1))
verificationW2 = norm(grad_W2-ngrad_W2,1)/max(1e-6,norm(grad_W2,1)+norm(ngrad_W2,1))
grad_b1 = grad_b{1};
grad_W1 = grad_W{1};
ngrad_b1 = ngrad_b{1};
ngrad_W1 = ngrad_W{1};
verificationb1 = norm(grad_b1-ngrad_b1,1)/max(1e-6,norm(grad_b1,1)+norm(ngrad_b1,1))
verificationW1 = norm(grad_W1-ngrad_W1,1)/max(1e-6,norm(grad_W1,1)+norm(ngrad_W1,1))%verify that this is small

%% Exercise 7
function Thetaopt = MiniBatchGD(Xtrain, Ytrain, Xdev, Ydev, GDparams, Theta, lambda)
    disp("Starting training of model")
    itermax = size(Theta,1);
    [~,N] = size(Xtrain);
    n_batch = GDparams.n_batch;
    n_s = GDparams.n_s;
    etamax = GDparams.etamax;
    etamin = GDparams.etamin;
    nCycles = GDparams.nCycles;
    eta = etamin;
    etahist = zeros(nCycles*n_s*2,1);
    etahist(1) = eta;
    n_epochs = GDparams.n_epochs;
    %loss = zeros(2,nCycles*n_s*2/10);
    %accuracy = zeros(2,nCycles*n_s*2/10);
    loss = zeros(2,n_epochs);
    accuracy = zeros(2,n_epochs);
    ind = linspace(1,N,N);
    [~,I] = max(Ytrain);
    Yacctrain = I;
    [~,I] = max(Ydev);
    Yaccdev = I;
    
    t = 1;
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

                activationVals = EvaluateClassifier(Xbatch,Theta);

                [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch,activationVals, Theta, lambda);
                
            for i = 1:itermax
                Theta{i,1} = Theta{i,1}-eta*grad_W{i};
                Theta{i,2} = Theta{i,2}-eta*grad_b{i};
            end
            
            etacal = mod(t,2*n_s);
            l = floor(t/(2*n_s));
            if etacal<n_s
                eta = etamin +(t-2*l*n_s)/n_s*(etamax-etamin);
            else
                eta = etamax -(t-(2*l+1)*n_s)/n_s*(etamax-etamin);
            end
            
            etahist(t) = eta;
            if 0 
            %if mod(t,10) == 0 
                loss(:,t/10) = [ComputeCost(Xtrain, Ytrain, Theta, lambda); ComputeCost(Xdev, Ydev, Theta, lambda)];
                accuracy(:,t/10) = [ComputeAccuracy(Xtrain, Yacctrain, Theta); ComputeAccuracy(Xdev, Yaccdev, Theta)];
                disp("Training loss is: "+loss(1,t/10)+" Accuracy is: "+accuracy(1,t/10)+" || Dev loss is: "+loss(2,t/10)+" Accuracy is: "+accuracy(2,t/10))
            end
            t = t+1;
            
        end
        loss(:,epoch) = [ComputeCost(Xtrain, Ytrain, Theta, lambda); ComputeCost(Xdev, Ydev, Theta, lambda)];
        accuracy(:,epoch) = [ComputeAccuracy(Xtrain, Yacctrain, Theta); ComputeAccuracy(Xdev, Yaccdev, Theta)];
        disp("Training loss is: "+loss(1,epoch)+" Accuracy is: "+accuracy(1,epoch)+" || Dev loss is: "+loss(2,epoch)+" Accuracy is: "+accuracy(2,epoch))
        if mod(epoch,5) == 0
            disp(epoch+"/"+n_epochs)
        end
    end
    
    figure(5)
    plot(etahist);
    figure(1)
    subplot(2,1,1)
    plot(1:epoch,loss(1,:),1:epoch,loss(2,:))
    legend('Training set','Dev set')
    title('Cost over epochs')
    xlabel('Epoch')
    ylabel('Cost')
    subplot(2,1,2)
    plot(1:epoch,accuracy(1,:),1:epoch,accuracy(2,:))
    legend('Training set','Dev set')
    title('Accuracy over epochs')
    xlabel('Epoch')
    ylabel('Accuracy')
    
    if 0
        figure(1)
        subplot(2,1,1)
        plot(1:floor(size(loss,2)/3),loss(1,1:floor(size(loss,2)/3)),1:floor(size(loss,2)/3),loss(2,1:floor(size(loss,2)/3)))
        legend('Training set','Dev set')
        title('Cost over epochs')
        xlabel('Epoch')
        ylabel('Cost')
        subplot(2,1,2)
        plot(1:floor(size(accuracy,2)/3),accuracy(1,1:floor(size(accuracy,2)/3)),1:floor(size(accuracy,2)/3),accuracy(2,1:floor(size(accuracy,2)/3)))
        legend('Training set','Dev set')
        title('Accuracy over epochs')
        xlabel('Epoch')
        ylabel('Accuracy')
        figure(6)
        subplot(2,1,1)
        plot(1:(size(loss,2)),loss(1,1:(size(loss,2))),1:(size(loss,2)),loss(2,1:(size(loss,2))))
        legend('Training set','Dev set')
        title('Cost over epochs')
        xlabel('Epoch')
        ylabel('Cost')
        subplot(2,1,2)
        plot(1:(size(accuracy,2)),accuracy(1,1:(size(accuracy,2))),1:(size(accuracy,2)),accuracy(2,1:(size(accuracy,2))))
        legend('Training set','Dev set')
        title('Accuracy over epochs')
        xlabel('Epoch')
        ylabel('Accuracy')
    end
    Thetaopt = Theta;
    disp("Training of model completed")
    
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
% ForwardPropogation
function activationVals = EvaluateClassifier(X,Theta)
    itermax = size(Theta,1);
    activationVals = {};
    a = X;
    for i = 1:itermax
        W = Theta{i,1};
        b = Theta{i,2};
        [~,N] = size(a);
        [K,~] = size(W);
        P = zeros(K,N);
        
        for j = 1:N
            s = W*a(:,j)+b;
            activationFunction = Theta{i,5};
            if  strcmp(activationFunction, 'softmax')
                P(:,j) = exp(s)/sum(exp(s))';
            else
                P(:,j) = max(0,s)';
            end
        end
        activationVals{i} = P;
        a = P;
    end
end

%% Exercise 4
% Calculate loss
function J = ComputeCost(X, Y, Theta, lambda)
    itermax = size(Theta,1);
    [~,N] = size(X);

    P = EvaluateClassifier(X,Theta);
    crossEntropy = 0;
    P = P{1,end};
    for i = 1:N
        crossEntropy = crossEntropy - log(Y(:,i)'*P(:,i));
    end
    regTerm = 0;
    for i = 1:itermax
        W = Theta{i,1};
        regTerm = regTerm+lambda*W(:)'*W(:);
    end
    J = crossEntropy/N+regTerm;
end
%% Exercise 5
% Calculate accurace. Uses labels for Y.
function acc = ComputeAccuracy(X, Y, Theta)
    [~,N] = size(X);
    P = EvaluateClassifier(X,Theta);
    P = P{1,end};
    [~,I] = max(P);
    acc = numel(find(I==Y))/N;
end

%% Exercise 6
% Calculate the gradient for backpropogation
function [grad_b, grad_W] = ComputeGradients(X, Y, activationVals, Theta, lambda)
    itermax = size(Theta,1);
    grad_b = {};
    grad_W = {};
    dJdz = 1;
    for i = itermax:-1:1
        z = activationVals{i};
        if i-1 == 0
            a = X;
        else
            a = activationVals{i-1};
        end
        [~,N] = size(z);
        activationFunction = Theta{i,5};
        % derivative of activation function
        if  strcmp(activationFunction, 'softmax')
            %Softmax
            dJdz = dJdz.*(z-Y);
        else
            %ReLU
            comp1 = max(0,z);
            comp2 = ones(size(comp1));
            dJdz = dJdz.*(comp1 & comp2);
        end
        grad_W{i} = dJdz*a'/N+lambda*2*Theta{i,1};
        grad_b{i} = sum(dJdz,2)/N;
        dJdz = Theta{i,1}'*dJdz;
        Y = z;
    end
end

function [W, b] = initParams(mean, std, N, M)
    W = std .* randn(N,M)+mean;
    b = zeros(N,1);
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

function [grad_b_cell, grad_W_cell] = ComputeGradsNum(X, Y, Theta, lambda, h)
    itermax = size(Theta,1);
    

    
    for j = itermax:-1:1
        W = Theta{j,1};
        b = Theta{j,2};
        no = size(W, 1);
        d = size(X, 1);
        grad_W = zeros(size(W));
        grad_b = zeros(no, 1);
        c = ComputeCost(X, Y, Theta, lambda);
        for i=1:length(b)
            b_try = b;
            b_try(i) = b_try(i) + h;
            ThetaTry = Theta;
            ThetaTry{j,2} = b_try;
            c2 = ComputeCost(X, Y, ThetaTry, lambda);
            grad_b(i) = (c2-c) / h;
        end

        for i=1:numel(W)   

            W_try = W;
            W_try(i) = W_try(i) + h;
            ThetaTry = Theta;
            ThetaTry{j,1} = W_try;
            c2 = ComputeCost(X, Y, ThetaTry, lambda);

            grad_W(i) = (c2-c) / h;
        end
        grad_b_cell{j} = grad_b;
        grad_W_cell{j} = grad_W;
    end
end