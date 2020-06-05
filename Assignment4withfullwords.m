clear; clc;



%%
Xchars = bookdata(1:params.seq_length);
Ychars = bookdata(2:params.seq_length+1);
X = string2indices(Xchars,char_to_ind);
Y = string2indices(Ychars,char_to_ind);

%% training loop
for j = 1:1
    [bookdata, char_to_ind, ind_to_char,N, RNN] = initModel();
    h = RNN.h0;
    Xtrainchars = bookdata(1:RNN.hyperparams.seq_length);
    Ytrainchars = bookdata(2:RNN.hyperparams.seq_length+1);
    trainX = string2indices(Xtrainchars,char_to_ind);
    trainY = string2indices(Ytrainchars,char_to_ind);
    ForwardPassVals = forwardpasscalc(RNN,h,trainX,trainY);
    smoothloss = ForwardPassVals.loss;
    e = 1;
    step = 1;
    
    
    itermax = floor(N*RNN.hyperparams.N_EPOCHS/RNN.hyperparams.seq_length);
    
    disp("========================================");
    disp("Iteration: "+1+"/"+itermax+" Smooth loss = " + smoothloss);
    disp("========================================");
    str1hot = generateMessage(RNN,trainX(:, 1) ,h,200);
    message = indices2string(str1hot,ind_to_char);
    disp(message);
    disp(" ");

    loss2print = [];

    
    for i = 1:floor(N*RNN.hyperparams.N_EPOCHS/RNN.hyperparams.seq_length)
        Xtrainchars = bookdata(e:e+RNN.hyperparams.seq_length-1);
        Ytrainchars = bookdata(e+1:e+RNN.hyperparams.seq_length);
        trainX = string2indices(Xtrainchars,char_to_ind);
        trainY = string2indices(Ytrainchars,char_to_ind);

        ForwardPassVals = forwardpasscalc(RNN,h,trainX,trainY);
        grads = backwardsprop(RNN,ForwardPassVals,trainX);

        for f = fieldnames(RNN.modelparams)'
            grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
            RNN.m.(f{1}) = RNN.m.(f{1})+grads.(f{1}).^2;
            RNN.modelparams.(f{1}) = RNN.modelparams.(f{1}) - RNN.hyperparams.eta./((RNN.m.(f{1})+eps).^(1/2)).* grads.(f{1});
        end
        smoothloss = .999* smoothloss + .001 * ForwardPassVals.loss;
        if mod(step,10) == 0
            if mod(step,100) == 0
                disp("========================================");
                disp("Iteration: "+step+"/"+itermax+" Smooth loss = " + smoothloss);
                disp("========================================");
                str1hot = generateMessage(RNN,trainX(:, 1) ,h,200);
                message = indices2string(str1hot,ind_to_char);
                disp(message);
                disp(" ");
            else
                disp("Iteration: "+step+"/"+itermax+" Smooth loss = " + smoothloss);
            end
        end
        if mod(step,10) == 0
            loss2print = [loss2print; smoothloss];
        end
        e = e+RNN.hyperparams.seq_length;
        if e > length(bookdata)-RNN.hyperparams.seq_length-1
            e = 1;
            h = zeros(size(h));
        else
            h = ForwardPassVals.h{end};
        end
        step = step+1;
    end
    figure(j)
    plot(loss2print)
    grid on
    save("RNN1"+j,"RNN")
end
%% Tests
clc
str1hot = generateMessage(RNN,x0,h0,25);
message = indices2string(str1hot,ind_to_char)

%% For report iv)
str1hot = generateMessage(RNN,X(:, 1),RNN.h0,1000);
message = indices2string(str1hot,ind_to_char)

%% One iteration
ForwardPassVals = forwardpasscalc(RNN,h0,X,Y);
grads = backwardsprop(RNN,ForwardPassVals,X);

for f = fieldnames(RNN.modelparams)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    RNN.modelparams.(f{1}) = RNN.modelparams.(f{1}) - RNN.hyperparams.eta * grads.(f{1});
end


%% Gradient Test
clc
ForwardPassVals = forwardpasscalc(RNN,h0,X,Y);
h = 1e-4;
grads = backwardsprop(RNN,ForwardPassVals,X);
numgrads = ComputeGradsNum(X, Y, RNN.modelparams, h);

gradV = grads.V(:);
numgradV = numgrads.V(:);
verification = norm(gradV-numgradV,1)/max(1e-6,norm(gradV,1)+norm(numgradV,1));
disp("Relative error is: "+verification)

%% BackPass
function grads = backwardsprop(RNN,Fwd,X)
    
    grads.V = Fwd.dLdo{Fwd.Tau}'*Fwd.h{Fwd.Tau}';
    grads.W = 0;
    
    
    dLdh{Fwd.Tau} = Fwd.dLdo{Fwd.Tau}*RNN.modelparams.V;
    dLda{Fwd.Tau} = dLdh{Fwd.Tau}*diag(1-tanh(Fwd.a{Fwd.Tau}).^2);
    
    grads.U = dLda{Fwd.Tau}'*X(:,Fwd.Tau)';
    
    grads.b = dLda{Fwd.Tau}';
    grads.c = Fwd.dLdo{Fwd.Tau}';
    
    for i = Fwd.Tau-1:-1:1
        dLdh{i} = Fwd.dLdo{i}*RNN.modelparams.V+dLda{i+1}*RNN.modelparams.W;
        dLda{i} = dLdh{i}*diag(1-tanh(Fwd.a{i}).^2);
        
        % Completed gradients
        grads.V = grads.V + Fwd.dLdo{i}'*Fwd.h{i}';
        grads.W = grads.W + dLda{i+1}'*Fwd.h{i}';
        grads.U = grads.U + dLda{i}'*X(:,i)';
        grads.b = grads.b + dLda{i}';
        grads.c = grads.c + Fwd.dLdo{i}';
    end
    
    %To get final step
    grads.W = grads.W + dLda{1}'*RNN.h0';
end

%% ForwardPass
function ForwardPassVals = forwardpasscalc(RNN,h0,X,Y)
    iterations = size(X,2);
    h = h0;
    loss = 0;
    for i = 1:iterations
        a = RNN.modelparams.W*h+RNN.modelparams.U*X(:,i)+RNN.modelparams.b;
        h = tanh(a);
        o = RNN.modelparams.V*h+RNN.modelparams.c;
        p = softmax(o);
        cost = -log(Y(:,i)'*p);
        loss = loss+cost;
        ForwardPassVals.a{i} = a;
        ForwardPassVals.h{i} = h;
        ForwardPassVals.o{i} = o;
        ForwardPassVals.p{i} = p;
        ForwardPassVals.y{i} = Y(:,i);
        ForwardPassVals.cost{i} = cost;
        ForwardPassVals.dLdo{i} = -(Y(:,i)-p)';
    end
    ForwardPassVals.loss = loss;
    ForwardPassVals.Tau = iterations;
end

%% Init model
function [bookdata, char_to_ind, ind_to_char,N, RNN] = initModel()
    [bookdata, char_to_ind, ind_to_char] = loadDataSet();
    N = size(bookdata,2);

    % Hyper parameters
    params.K = char_to_ind.size(1);
    params.m = 400;
    params.sig = .01;
    params.eta = .05;
    params.seq_length = 25;
    params.N_EPOCHS = 3;

    % Initialize model
    RNN.hyperparams = params;
    RNN.modelparams.b = zeros(params.m,1);
    RNN.modelparams.c = zeros(params.K,1);
    RNN.modelparams.U = randn(params.m, params.K)*params.sig;
    RNN.modelparams.W = randn(params.m, params.m)*params.sig;
    RNN.modelparams.V = randn(params.K, params.m)*params.sig;
    RNN.m.b = 0;
    RNN.m.c = 0;
    RNN.m.U = 0;
    RNN.m.W = 0;
    RNN.m.V = 0;

    x0 = zeros(params.K,1);
    idx = randi([0 params.K]);
    x0(idx) = 1;
    h0 = zeros(params.m,1);
    RNN.h0 = h0;

end
%% NiceToHaveFunctions

function probs = softmax(vec)
    num = exp(vec);
    den = sum(num,1);
    probs = num./den;
end
function [bookdata,char_to_ind,ind_to_char] = loadDataSet()
    path = matlab.desktop.editor.getActiveFilename;
    [filepath,~,~] = fileparts(path);
    addpath(filepath+ "/Datasets/");
    book_fname = filepath+"/Datasets/goblet_book.txt";
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c');
    fclose(fid);
    
    [book_dataprime,matches] = split(lower(book_data),[" ","!",",",".","""","?","(",")","&",newline,char(9)]);
    book_data = {};
    for i = 1:size(matches,1)
        %book_data = [book_data;book_dataprime(i);matches(i)];
        book_data(2*i-1) = book_dataprime(i);
        book_data(2*i-0) = matches(i);
    end
    book_data(end) = book_dataprime(end);
    
    [C,IA,IC] = unique(book_data);
    K = size(C,2);
    character = {};
    for i = 1:K
        characters{i} = C(i);
    end

    char_to_ind = containers.Map(C,linspace(1,K,K));
    ind_to_char = containers.Map(linspace(1,K,K),C);
    bookdata = book_data;
end
function str1hot = generateMessage(RNN,x0,h0,n)
    x = x0;
    d = size(x0,1);
    tau = size(x0,1)-1;
    h = h0;
    str1hot = zeros(d,n);
    for i = 1:n
        a = RNN.modelparams.W*h+RNN.modelparams.U*x+RNN.modelparams.b;
        h = tanh(a);
        o = RNN.modelparams.V*h+RNN.modelparams.c;
        p = softmax(o);
        
        
        cp = cumsum(p);
        ixs = find(cp-rand >0);
        x = zeros(d,1);
        x(ixs(1)) = 1;
        str1hot(:,i) = x;
    end
end
function message = indices2string(indeces,ind_to_char)
    n = size(indeces,2);
    message = "";
    [v,idx] = max(indeces);
    for i = 1:n
        character = ind_to_char(idx(i));
        message = message+character;
    end
end
function indeces = string2indices(str,char_to_ind)
    n = size(str,2);
    d = char_to_ind.size(1);
    indeces = zeros(d,n);
    for i = 1:n
        idx = char_to_ind(str{i});
        indeces(idx,i) = 1;
    end
end
