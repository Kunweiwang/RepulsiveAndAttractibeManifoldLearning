%% load the data and creat the result fold.
clear;
epsi=0.05;K1=4;K2=6;
dir_name='E:\\stemp\\SCUT-FBP\\';
result_file = 'E:\\stemp\\SCUT-FBP\\Manifoldlearning';
load('E:\\stemp\\SCUT-FBP\\initial_data_SCUT_vgg.mat');%label data
load('E:\\stemp\\SCUT-FBP\\Distance_pca_7.mat');%images' distances between other images.
load('E:\\stemp\\SCUT-FBP\\Distance_scores.mat');%images score distances between other images.

if ~exist('E:\\stemp\\SCUT-FBP\\Manifoldlearning','dir')
    mkdir(result_file);
end
%% Split the 500 images into five parts
CV1=cvpartition(labels0,'KFold',5);

%%
Dist_s = pdist2(labels0,labels0);

Dist_s(Dist_s>0.05)=101;
Dist_s(logical(eye(500)))=100;
Dist(logical(eye(500)))=inf;
[Dist_sort, sort_inx]= sort(Dist_s,2);

Ne_all = zeros(500,500);
Nr_all = zeros(500,500);



for i = 1:500
    imagei = find(Dist_s(i,:)==100);
    Ne_all(i,1:imagei-1) = sort_inx(i,1:imagei-1);
    Nr_all(i,1:500-imagei) = sort_inx(i,imagei+1:500);
end

Xpca_7= double(Xpca_7);
W_matrix = zeros(200,CV1.NumTestSets);
Pred_score = zeros(1,100);
MAE = zeros(1,5);
Pearson = zeros(1,5);
 for i = 1:CV1.NumTestSets
          trIdx = find(CV1.training(i));
          teIdx = find(CV1.test(i));
          Na = zeros(400,K1);
          Nr = zeros(400,K2);
          A = zeros(200,200);
          B = zeros(200,200);
          for ii = 1:length(trIdx)
              Ne_inter = intersect(Ne_all(trIdx(ii),:),trIdx);
              [distance_ii,distance_sort] = sort(Dist(trIdx(ii),Ne_inter));
              if length(distance_sort)>=K1
                  Na(ii,:) = distance_sort(1:K1);
                  for j = Na(ii,:)
                      A = A + (Xpca_7(:,trIdx(ii))-Xpca_7(:,j))*(Xpca_7(:,trIdx(ii))-Xpca_7(:,j))';
                  end
              else
                  Na(ii,1:length(distance_sort)) = distance_sort;
                  for j = distance_sort
                      A = A + (Xpca_7(:,trIdx(ii))-Xpca_7(:,j))*(Xpca_7(:,trIdx(ii))-Xpca_7(:,j))';
                  end
              end
              
              Nr_inter = intersect(Nr_all(trIdx(ii),:),trIdx);
              [distance_Nr,Nr_sort] = sort(Dist(trIdx(ii),Nr_inter));
              if length(Nr_sort)>=K2
                  Nr(ii,:) = Nr_sort(1:K2);
                  for j = Nr(ii,:)
                      B = B + (Xpca_7(:,trIdx(ii))-Xpca_7(:,j))*(Xpca_7(:,trIdx(ii))-Xpca_7(:,j))';
                  end
              else
                  Nr(ii,1:length(Nr_sort)) = Nr_sort;
                  for j = Nr_sort
                      B = B + (Xpca_7(:,trIdx(ii))-Xpca_7(:,j))*(Xpca_7(:,trIdx(ii))-Xpca_7(:,j))';
                  end
              end
          end
          [m_eig,value] =eig(A^-1*B);
          m = find(max(diag(value)));
          W = m_eig(:,m);
          w_matrix(:,i) = W;
          
          prediction = W'*Xpca_7(:,teIdx);
          mae_result = sum(abs(prediction-labels0(teIdx,1)'))/100;
          pearson_result = corr(prediction',labels0(teIdx,1));    
          MAE(i) = mae_result;
          Pearson(i) = pearson_result;
 end
result_name = strcat(result_file,'\\W_matrix.mat'); 
save(result_name,'W_matrix','MAE','Pearson');
