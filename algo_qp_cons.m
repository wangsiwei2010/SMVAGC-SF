function [UU,A,Z,iter,obj,alpha,U] = algo_qp_cons(X,Y,lambda,d,numanchor)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);

M = cell(numview,1);
p = zeros(numclass,numsample);
p(:,1:numclass) = eye (numclass);
[Z, ~, ~]=svd(p','econ');
Z = Z';

for i = 1:numview
   m = numanchor(i);
   A{i} = initialize(X{i},m);
   A{i} = A{i}';
   X{i} = mapstd(X{i}',0,1); % turn into d*n

   M{i} = (A{i}'*X{i}); 
   for ii=1:numsample
       idx = 1:m;
       pp(idx,ii) = EProjSimplex_new(M{i} (idx,ii));           
    end
    Zi{i} = pp;
    clear pp;
end

alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize A_i
    parfor ia = 1:numview
        C = X{ia}*Zi{ia}';      
        [U,~,V] = svd(C,'econ');
        A{ia} = U*V';
    end
    
     %% optimize Ri
    for j=1:numview
        [Unew1,~,Vnew1] = svd(Z*Zi{j}','econ');
        R{j} = Unew1*Vnew1';
    end
    
   %% optimize Z-i            

    for a=1:numview
        M{a} = (alpha(a)^2*A{a}'*X{a}+ lambda *R{a}' *Z)/(alpha(a)^2 + lambda); 
        for ii=1:numsample
            idx = 1:numanchor(a);
            pp(idx,ii) = EProjSimplex_new(M{a} (idx,ii));           
        end
        Zi{a} = pp;
        clear pp;
    end
    
    %% optimize Z
    part1 = 0;
    for ia = 1:numview
        part1 = part1 + R{ia} *Zi{ia};
    end
    [Unew,~,Vnew] = svd(part1,'econ');
    Z = Unew*Vnew';
   
    %% optimize alpha
    P = zeros(numview,1);
    for iv = 1:numview
        P(iv) = norm( X{iv} - A{iv} * Zi{iv},'fro')^2;
    end
    Mfra = P.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    term2 =0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - A{iv} * Zi{iv},'fro')^2;
        term2 = term2 + lambda * norm(R{iv} *Zi{iv} - Z,'fro')^2;
    end

    obj(iter) = term1+ term2;
    
    U{iter}=Z';
     
    
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6 || iter>maxIter || obj(iter) < 1e-10)
        UU = Z'; 
        flag = 0;
    end
end
         
         
    
