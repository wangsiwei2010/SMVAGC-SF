 clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'NGs'};
dsPath = './dataset/';
resPath = './res-lmd/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1:1:1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         :length(ds)
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName)); 
    k = length(unique(Y));
    numofview = length(X);
    
    
    matpath = strcat(resPath,dataName);
    txtpath = strcat(resPath,strcat(dataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    
    %% para setting
    selectanchor = [1,2,5]*k;
    anchormatrix = constructanchor(numofview,selectanchor);
    d = (1)*k ;
    lambda = 10.^[0:1:3];
    [n,~] = size(anchormatrix);
    
    %%
    for ichor = 1:length(anchormatrix)
        for id = 1:length(lambda)
            tic;
            [U,A,Z,iter,obj,alpha,P] = algo_qp_cons(X,Y,lambda(id),d,anchormatrix(ichor,:)); 
            U = U ./ repmat(sqrt(sum(U .^ 2, 2)), 1, k);
            
            [res std] = myNMIACCwithmean(U,Y,k); 
            timer(ichor,id)  = toc;
            fprintf('Anchor:%d \t Lambda:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[ichor lambda(id) res(1) res(2) res(3) res(4) timer(ichor,id)]);
            
            resall{ichor,id} = res;
            stdall{ichor,id} = std;
            objall{ichor,id} = obj;
            
            dlmwrite(txtpath, [ichor lambda(id) res std timer(ichor,id)],'-append','delimiter','\t','newline','pc');
            matname = ['_Anch_',num2str(anchormatrix(ichor)),'_Dim_',num2str(lambda(id)),'.mat'];

            save([resPath,'All_',dataName,'.mat'],'resall','stdall','objall','metric','anchormatrix');
        end
    end
    clear resall objall;
end


