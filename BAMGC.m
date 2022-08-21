function [index, Yp, Z, S, W,obj] = BAMGC(X, alpha, beta, sigma, group_num, group_label, maxIter,class_num)
%% input: 
%       dataset X (r times mn, r for #bands, mn for #samples)
%       hyper parameters alpha, beta, sigma
%       the number of group: view_num
%       the label of group: view_label
%       the number of iteration: maxIter
%  output:
%       feature selection index: index
%       (for features) self-representation matrix: Z
%       (for samples) global similarity S
%       (for groups) weight matrix: W
%       (for features) Mapping matrix Yp
% author: u

%% Initialize input parameters
[r, mn] = size(X);
c = class_num;
W = ones(group_num, mn)./group_num;
Z = rand(r,c);
S = zeros(mn,mn);
Yp = zeros(mn,c);
obj = zeros(maxIter, 1);

%% Initialize similarity matrix for each view
disp("Initialization complete");
S_temp = S;
for v = 1:group_num
    indx = group_label==v;
    data_v = X(indx,:);
    for i = 1:mn
        for j = 1:mn
            S_temp(i,j) = exp(-(sum((data_v(:,i) - data_v(:,j)).^2))/sigma);
        end
        S_temp(:,i) = S_temp(:,i)./sum(S_temp(:,i));
    end
    S_d.data{v}=(S_temp+S_temp')./2;
end
clear S_temp indx data_v;
for i = 1:mn
    for j = 1:mn
        S(i,j) = exp(-(sum((X(:,i) - X(:,j)).^2))/sigma);
    end
    S(:,i) = S(:,i)./sum(S(:,i));
end
S=(S+S')./2;

for iter = 1 : maxIter
    disp(iter);
    %% update W
    for i = 1:mn
        A_i = zeros(mn, group_num);
        for v = 1:group_num
            A_i(:,v) = S(:,i)-S_d.data{v}(:,i);
        end
        part_bi = A_i'*A_i;
        part_1v = ones(group_num,1);
%         temp_inv = part_bi \ part_1v;
        temp_inv = lsqminnorm(part_bi,part_1v);
        W(:,i) = temp_inv / (part_1v' * temp_inv + 1e-15);
    end
    clear A_i part_bi part_1v;
    
    %% update S
    for i = 1:mn
        B_i = zeros(mn, group_num);
        for v = 1:group_num
            B_i(:,v) = S_d.data{v}(:,i);
        end
        a_i = zeros(mn, 1);
        for p = 1:mn
            a_i(p) = 0.5* beta * norm(Yp(i,:)-Yp(p,:), 'fro')^2;
        end
        c_i = 2 * B_i * W(:,i) - a_i;
        psi = zeros(mn, 1);
        temp_1= 0.5 *(eye(mn) - 1 / mn * ones(mn,mn)) * c_i + 1 / mn * ones(mn,1) - 0.5 * mean(psi) * ones(mn,1);
        for j = 1:mn
%             temp_1= 0.5 *(eye(mn) - 1 / mn * ones(mn,mn)) * c_i + 1 / mn * ones(1,mn) - 0.5 * mean(psi);
            psi(j) = max(-2*temp_1(j), 0);
        end
        temp_1= 0.5 *(eye(mn) - 1 / mn * ones(mn,mn)) * c_i + 1 / mn * ones(mn,1) - 0.5 * mean(psi) * ones(mn,1);
        for j = 1:mn
            
            
%             temp_1= 0.5 *(eye(mn) - 1 / mn * ones(mn,mn)) * c_i + 1 / mn * ones(1,mn) - 0.5 * mean(psi);
            S(i, j) = max(temp_1(j), 0);
        end
    end
    clear B_i a_i c_i psi temp_1;
    
    %% update Z
    for loop = 1 : maxIter
        temp = 2 * sqrt(sum(Z.^2, 2)) + 1e-15;
        Q = diag(1./temp);
        temp1 = X * X' + alpha * Q;
%         Z = temp1 \ X * Yp;      关于矩阵接近奇异值，或放缩错误。结果可能不准确。
        Z = lsqminnorm(temp1,X) * Yp;
        clear temp Q temp1;
    end
    
    %% update Yp
%     for loop = 1 : maxIter
        LapMatrix = diag(sum(S, 2)) - S;
        temp_A = eye(mn) + beta * LapMatrix;
        temp_lamuda = max(diag(temp_A));
        temp_M = (temp_lamuda * eye(mn) - temp_A ) * Yp + 2 * X' * Z;
        [svd_U,~,svd_V] = svd(temp_M, 'econ');
        Yp = svd_U * eye(c) * svd_V';
        clear svd_V svd_U temp_A temp_formulationYp2 temp_lamuda temp_M LapMatrix;
%     end
    
    %% calculate objective function value1
    temp_formulation1 = 0;
    for i =1:mn
        temp_S_j = zeros(mn,1);
        for v = 1:group_num
            temp_S_j = temp_S_j + W(v,i)*S_d.data{v}(:,i);
        end
        temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_j,'fro')^2;
    end
    LapMatrix = diag(sum(S, 2)) - S;
    temp_formulation2 = norm(X'*Z-Yp,'fro')^2;
    temp_formulation3 = alpha * sum(sqrt(sum(Z.^2, 2)));%L2,1-norm
    temp_formulation4 = beta * trace(Yp'*LapMatrix * Yp);
    obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3 + temp_formulation4;
    clear LapMatrix temp_formulation1 temp_S_j temp_S_i temp_formulation2 temp_formulation3 temp_formulation4;
    if iter>1
        err = abs(obj(iter - 1) - obj(iter));
        disp(err);
    end
    clear err
end

score=sum((Z.*Z),2);  %Z.*Z r*c  按行求和(关于c求和)%%修改
[~,index]=sort(score,'descend');

end