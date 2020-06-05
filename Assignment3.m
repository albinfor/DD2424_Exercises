%% Initialize data with all variables
clear all; clc; close all;

%Design Params
lambda =  0.0035;
GDparams.nCycles = 3;
GDparams.n_batch = 100;
GDparams.etamin = 1e-5;
GDparams.etamax = 1e-1;
GDparams.alpha = 0.9;

%Run settings (Use all zeros for "vanilla" run)
usemoredata = 1;
GDparams.ordershuffle = 1;
doPCA = 0;

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
    
    [trainX,trainY1hot,trainY, devX, devY1hot, devY] = traindevsplit(90, X, Y1hot, Y);
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

if doPCA
    noComponents = 1000;
    C = cov(trainX');
    [V,D] = eig(C);
    trainX = V(:,end-noComponents+1:end)' * trainX;
    devX = V(:,end-noComponents+1:end)' * devX;
    testX = V(:,end-noComponents+1:end)' * testX;
end

[~,N] = size(trainX);

%Calculates n
%GDparams.n_s = 2*floor(N/GDparams.n_batch);
GDparams.n_s = 5*floor(N/GDparams.n_batch);
GDparams.n_s = 5*floor(45000/GDparams.n_batch);
GDparams.n_epochs = ceil(2*GDparams.n_s*GDparams.nCycles*GDparams.n_batch/N);

% Choose network size
k = [50,50];
Theta = GetModel(k,trainX,trainY1hot);

%% Run a normal training session
% Train and evaluate model
[Thetaopt, mu, var]= MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, Theta, lambda);
acctrain = ComputeAccuracy(trainX, trainY, Thetaopt,mu,var);
disp("Accuracy on the train data is: " + acctrain);

acctest = ComputeAccuracy(testX, testY, Thetaopt,mu,var);
disp("Accuracy on the test data is: " + acctest);

%Display trained models
Layers = size(Theta,1);
if not(doPCA)
    W = Thetaopt{1,1};
    gamma = Thetaopt{1,6};
    W = W.*gamma;
    [M,~] = size(W);
    for i=1:M
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
figure(2)
montage(s_im);
end

%% Look for good Lambda
l_min = -3;
l_max = -2;
n_samples = 40;
accuracies = [];
Lambdas = [];
for j = 1:n_samples
    %l = l_min + (l_max - l_min)*rand(1, 1);
    %Lambda = 10^l;
    l = linspace(l_min,l_max,n_samples);
    Lambda = 10^l(j);
    disp("Training run: "+j+", Lambda = "+Lambda);
    [Thetaopt, mu, var] = MiniBatchGD(trainX, trainY1hot, devX, devY1hot, GDparams, Theta, Lambda);
    accdev = ComputeAccuracy(devX, devY, Thetaopt, mu, var);
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


%% Verify Gradient
clc;
ThetaTest = Theta;
W = ThetaTest{1,1};
ThetaTest{1,1} = W(:, 1:20);

Ptest = EvaluateClassifier(trainX(1:20, 1:10),ThetaTest);
[grad_b, grad_W, grad_beta, grad_gamma] = ComputeGradients(trainX(1:20, 1:10), trainY1hot(:, 1:10),Ptest, ThetaTest, lambda);
[ngrad_b, ngrad_W, ngrad_beta, ngrad_gamma] = ComputeGradsNum(trainX(1:20, 1:10), trainY1hot(:, 1:10), ThetaTest, lambda, 1e-6);

%% third layer
grad_b3 = grad_b{3};
grad_W3 = grad_W{3};
grad_beta3 = grad_beta{3};
grad_gamma3 = grad_gamma{3};
ngrad_b3 = ngrad_b{3};
ngrad_W3 = ngrad_W{3};
ngrad_beta3 = ngrad_beta{3};
ngrad_gamma3 = ngrad_gamma{3};
verificationb3 = norm(grad_b3-ngrad_b3,1)/max(1e-6,norm(grad_b3,1)+norm(ngrad_b3,1));
verificationW3 = norm(grad_W3-ngrad_W3,1)/max(1e-6,norm(grad_W3,1)+norm(ngrad_W3,1));
verificationbeta3 = norm(grad_beta3-ngrad_beta3,1)/max(1e-6,norm(grad_beta3,1)+norm(ngrad_beta3,1));
verificationgamma3 = norm(grad_gamma3-ngrad_gamma3,1)/max(1e-6,norm(grad_gamma3,1)+norm(ngrad_gamma3,1));
verification3 = norm([verificationb3,verificationW3,verificationgamma3,verificationbeta3])

%% second layer
grad_b2 = grad_b{2};
grad_W2 = grad_W{2};
grad_beta2 = grad_beta{2};
grad_gamma2 = grad_gamma{2};
ngrad_b2 = ngrad_b{2};
ngrad_W2 = ngrad_W{2};
ngrad_beta2 = ngrad_beta{2};
ngrad_gamma2 = ngrad_gamma{2};
verificationb2 = norm(grad_b2-ngrad_b2,1)/max(1e-6,norm(grad_b2,1)+norm(ngrad_b2,1));
verificationW2 = norm(grad_W2-ngrad_W2,1)/max(1e-6,norm(grad_W2,1)+norm(ngrad_W2,1));
verificationbeta2 = norm(grad_beta2-ngrad_beta2,1)/max(1e-6,norm(grad_beta2,1)+norm(ngrad_beta2,1));
verificationgamma2 = norm(grad_gamma2-ngrad_gamma2,1)/max(1e-6,norm(grad_gamma2,1)+norm(ngrad_gamma2,1));
verification2 = norm([verificationb2,verificationW2,verificationgamma2,verificationbeta2])

%% first layer
grad_b1 = grad_b{1};
grad_W1 = grad_W{1};
grad_beta1 = grad_beta{1};
grad_gamma1 = grad_gamma{1};
ngrad_b1 = ngrad_b{1};
ngrad_W1 = ngrad_W{1};
ngrad_beta1 = ngrad_beta{1};
ngrad_gamma1 = ngrad_gamma{1};
verificationb1 = norm(grad_b1-ngrad_b1,1)/max(1e-6,norm(grad_b1,1)+norm(ngrad_b1,1));
verificationW1 = norm(grad_W1-ngrad_W1,1)/max(1e-6,norm(grad_W1,1)+norm(ngrad_W1,1));
verificationbeta1 = norm(grad_beta1-ngrad_beta1,1)/max(1e-6,norm(grad_beta1,1)+norm(ngrad_beta1,1));
verificationgamma1 = norm(grad_gamma1-ngrad_gamma1,1)/max(1e-6,norm(grad_gamma1,1)+norm(ngrad_gamma1,1));
verification1 = norm([verificationb1,verificationW1,verificationgamma1,verificationbeta1])

%% Exercise 7
function [Thetaopt, mu_av, v_av] = MiniBatchGD(Xtrain, Ytrain, Xdev, Ydev, GDparams, Theta, lambda)
    disp("Starting training of model")
    itermax = size(Theta,1);
    [~,N] = size(Xtrain);
    alpha = GDparams.alpha;
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
    mu_av = {};
    v_av = {};
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
            
            [grad_b, grad_W, grad_beta, grad_gamma] = ComputeGradients(Xbatch, Ybatch,activationVals, Theta, lambda);
                
            for i = 1:itermax
                Theta{i,1} = Theta{i,1}-eta*grad_W{i};
                Theta{i,2} = Theta{i,2}-eta*grad_b{i};
                Theta{i,6} = Theta{i,6}-eta*grad_gamma{i};
                Theta{i,7} = Theta{i,7}-eta*grad_beta{i};
                
                %Update moving average
                mu = activationVals{i,4};
                v = activationVals{i,5};
                if t == 1
                    mu_av(i) = {mu};
                    v_av(i) = {v};
                else
                    mu_av(i) = {alpha*mu_av{i}+(1-alpha)*mu};
                    v_av(i) = {alpha*v_av{i}+(1-alpha)*v};
                end
            end
            
            etacal = mod(t,2*n_s);
            l = floor(t/(2*n_s));
            if etacal<n_s
                eta = etamin +(t-2*l*n_s)/n_s*(etamax-etamin);
            else
                eta = etamax -(t-(2*l+1)*n_s)/n_s*(etamax-etamin);
            end
            
            etahist(t) = eta;
            t = t+1;

        end
        loss(:,epoch) = [ComputeCost(Xtrain, Ytrain, Theta, lambda,mu_av,v_av); ComputeCost(Xdev, Ydev, Theta, lambda,mu_av,v_av)];
        accuracy(:,epoch) = [ComputeAccuracy(Xtrain, Yacctrain, Theta,mu_av,v_av); ComputeAccuracy(Xdev, Yaccdev, Theta,mu_av,v_av)];
        
        disp("Training loss is: "+loss(1,epoch)+" Accuracy is: "+accuracy(1,epoch)+" || Dev loss is: "+loss(2,epoch)+" Accuracy is: "+accuracy(2,epoch))
        if mod(epoch,5) == 0
            disp(epoch+"/"+n_epochs)
        end
    end
    
    figure(5)
    plot(etahist);
    figure(1)
    subplot(2,1,1)
    plot(1:n_epochs,loss(1,:),1:n_epochs,loss(2,:))
    legend('Training set','Dev set')
    title('Cost over epochs')
    xlabel('Epoch')
    ylabel('Cost')
    subplot(2,1,2)
    plot(1:n_epochs,accuracy(1,:),1:n_epochs,accuracy(2,:))
    legend('Training set','Dev set')
    title('Accuracy over epochs')
    xlabel('Epoch')
    ylabel('Accuracy')
    Thetaopt = Theta;
    disp("Training of model completed")
end

%% Generate Theta
function Theta = GetModel(k,X,Y)
[d,~] = size(X);
[K,~] = size(Y);
k = [d k K];
[~,a] = size(k);
Theta = {};
sig = 1e-4; %Set sigma for all layers

for i = 1:(a-1)
    if i < a-1
        afunc = 'ReLU';
    else
        afunc = 'softmax';
    end
    [W, b] = initParams(0, 1/sqrt(k(i)), k(i+1), k(i));
    [W, b] = initParams(0, sig, k(i+1), k(i));
    [gamma, beta] = initParams(0, 2/sqrt(k(i+1)), k(i+1), 1);
    gamma = ones(k(i+1),1);
    beta = b;
    Theta(i,:) = {W,b,k(i+1),k(i),afunc,gamma,beta};
end

end

function s_hat = BatchNormalize(s,mu,v)
    s_hat = diag(v+eps)^(-1/2)*(s-mu);  
end

function G = BatchNormBackPass(G,s,mu,v)
    [~,N] = size(s);
    sigma1 = (v+eps).^(-1/2);
    sigma2 = (v+eps).^(-3/2);
    G1 = G.*(sigma1*ones(1,N));
    G2 = G.*(sigma2*ones(1,N));
    D = s-mu*ones(1,N);
    c = (G2.*D)*ones(N,1);
    G = G1-1/N*(G1*ones(N,1))*ones(1,N)-1/N*D.*(c*ones(1,N));
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
function activationVals = EvaluateClassifier(X,Theta,varargin)
    itermax = size(Theta,1);
    activationVals = {};
    a = X;
    if size(varargin,1) > 0
        mu_av = varargin{1};
        v_av = varargin{2};
    end
    for i = 1:itermax
        W = Theta{i,1};
        b = Theta{i,2};
        gamma = Theta{i,6};
        beta = Theta{i,7};
        [~,N] = size(a);
        [K,~] = size(W);
        P = zeros(K,N);
        activationFunction = Theta{i,5};
        
        s = W*a+b;
        if size(varargin,1) == 0
            m = size(s,2);
            v = var(s, 0, 2);
            v = v*((m-1)/m);
            mu = mean(s,2);
        else
            mu = mu_av{i};
            v = v_av{i};
        end
        if i == itermax
            s_hat = s;
            s_tilde = s;
        else
            s_hat = BatchNormalize(s,mu,v);
            s_tilde = gamma.*s_hat+beta;
        end
        if  strcmp(activationFunction, 'softmax')
            P = exp(s_tilde)./sum(exp(s_tilde),1);
        else
            P = max(0,s_tilde);
        end
        activationVals(i,:) = {P, s, s_hat,mu,v};
        a = P;
    end
end

%% Exercise 4
% Calculate loss
function J = ComputeCost(X, Y, Theta, lambda, varargin)
    itermax = size(Theta,1);
    [~,N] = size(X);
    if size(varargin,1) == 0
        P = EvaluateClassifier(X,Theta);
    else
        mu = varargin{1};
        var = varargin{2};
        P = EvaluateClassifier(X,Theta,mu,var);
    end
    crossEntropy = 0;
    P = P{end,1};
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
function acc = ComputeAccuracy(X, Y, Theta,mu,var)
    [~,N] = size(X);
    P = EvaluateClassifier(X,Theta,mu,var);
    P = P{end,1};
    [~,I] = max(P);
    acc = numel(find(I==Y))/N;
end

%% Exercise 6
% Calculate the gradient for backpropogation
function [grad_b, grad_W, grad_beta, grad_gamma] = ComputeGradients(X, Y, activationVals, Theta, lambda)
    itermax = size(Theta,1);
    grad_b = {};
    grad_W = {};
    grad_beta = {};
    grad_gamma = {};
    dJdz = 1;
    for i = itermax:-1:1
        z = activationVals{i,1};
        s = activationVals{i,2};
        s_hat = activationVals{i,3};
        mu = activationVals{i,4};
        v = activationVals{i,5};
        [D,N] = size(z);
        if i-1 == 0
            a = X;
        else
            a = activationVals{i-1,1};
        end
        
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
        
        if i < itermax
            grad_beta{i} = (1/N*dJdz*ones(N,1));
            grad_gamma{i} = (1/N*(dJdz.*s_hat)*ones(N,1));
            dJdz = 1/N.*Theta{i,6}.*(v+eps).^(-1/2).*(N.*dJdz-sum(dJdz,2)-(s-mu).*((v+eps).^(-1)).*sum(dJdz.*(s-mu),2));
            %dJdz = dJdz.*(Theta{i,6}*ones(1,N));
            %dJdz = BatchNormBackPass(dJdz,s,mu,v);
        else
            grad_beta{i} = 0;
            grad_gamma{i} = 0;
        end
        
        grad_W{i} = dJdz*a'/N+lambda*2*Theta{i,1};
        grad_b{i} = dJdz*ones(N,1)/N;
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
function [grad_b_cell, grad_W_cell, grad_beta_cell, grad_gamma_cell] = ComputeGradsNum(X, Y, Theta, lambda, h)
    itermax = size(Theta,1);
    

    
    for j = itermax:-1:1
        W = Theta{j,1};
        b = Theta{j,2};
        gamma = Theta{j,6};
        beta = Theta{j,7};
        no = size(W, 1);
        d = size(X, 1);
        grad_W = zeros(size(W));
        grad_b = zeros(no, 1);
        grad_gamma = zeros(no, 1);
        grad_beta = zeros(no, 1);
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
        
        for i=1:numel(beta)   

            beta_try = beta;
            beta_try(i) = beta_try(i) + h;
            ThetaTry = Theta;
            ThetaTry{j,7} = beta_try;
            c2 = ComputeCost(X, Y, ThetaTry, lambda);

            grad_beta(i) = (c2-c) / h;
        end
        
        for i=1:numel(gamma)   

            gamma_try = gamma;
            gamma_try(i) = gamma_try(i) + h;
            ThetaTry = Theta;
            ThetaTry{j,6} = gamma_try;
            c2 = ComputeCost(X, Y, ThetaTry, lambda);

            grad_gamma(i) = (c2-c) / h;
        end
        
        
        grad_b_cell{j} = grad_b;
        grad_W_cell{j} = grad_W;
        grad_beta_cell{j} = grad_beta;
        grad_gamma_cell{j} = grad_gamma;
    end
end