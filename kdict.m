function [D, C] = kdict(X, M, K)
image_data = double(X); %convert the data into double
x = x_matrix(image_data); %calls function x_matrix that takes 8x8 patches of the image and converts them into 64x1
D = randn(64,M); %generate a dictionary using random numbers
for k = 1:256 %normalise each column
    D(:,k) = D(:,k)/norm(D(:,k));
end
for j = 1:10 % number of iterations for the dictionary learning 
    C = OMP(K, x, D); %sparse coding
    [D, C] = dict_learning(x,C,D); %dictionary learning
end
end

% Reshape each patch of 8X8 into a vector of 64X1
% Returns a 64X1024 matrix
function f = x_matrix(data)
dim = size(data);
f = [];
n = 8; %dimension of the small matrix is n x n 
k = dim(1)/n; %
for j = 0:k-1
    for i = 0:k-1
        b = data(1+j*n:n+j*n,1+i*n:n+i*n);
        d = reshape(b,[],1);
        f = [f d];
    end
end
end

%OMP algorithm
%function OMP puts together all the columns of the sparse matrix
function [sparse_matrix] = OMP(sparsity, data, dict)
s = sparsity;
sparse_matrix = [];
for k = 1:1024 %for each column of the data matrix, do sparse coding
    x = data(:,k);
    s_index = zeros(s,1);
    max_val = zeros(s,1);
    dict_col = [];
    c = zeros(256,1);
    r = x;
    for i = 1:s
        c_temp = zeros(256,1);
        idx = setdiff([1:256],s_index);
        for j = idx % find the dot product of the data column and the respective atom to find the max value
            c_temp(j) = dot(dict(:,j),r);
        end
        max_val(i) = max(abs(c_temp)); % find the max value
        a = find(abs(c_temp)==max_val(i)); % find the respective index
        s_index(i) = a(1); % store the index value (in case there are multiple, take the first one)
        dict_col = [dict_col dict(:,s_index(i))]; % store the dictionary col associated with the index
        max_val = mldivide(dict_col,x); % calculate the coefficients in the sparse representation  
        r = x - dict_col*max_val; % build the new residual
    end
    for l = 1:s
        c(s_index(l)) = max_val(l); %final coefficient values 
    end
    sparse_matrix = [sparse_matrix c]; %put together the sparse representation for each data column
end
end

%Dictionary Learning
%this function creates E_k 
function [d, c] = dict_learning(x,c,d)
for i = 1:256
    E_k = x;
    idx = setdiff([1:256],i);
    for j = idx %subtract contributions from all the other atoms
        E_k = E_k - d(:,j)*c(j,:);
    end
    [d(:,i), c(i,:)] = dict_col(E_k, d(:,i), c(i,:));
end
end

%this functions does the updating of the dict and coefficients
function [d, c] = dict_col(E_k, d, c)
support = find(c); %indices with non-zero values  
if isempty(support) ~= 1 %check in case there are no non-zero values
    support_size = size(support);
    c_omega = zeros(support_size(2),1);
    E_komega = zeros(64,support_size(2));
    for i = 1:support_size(2)
        c_omega(i,1) = c(support(i));
        E_komega(:,i) = E_k(:,support(i));
    end
    [U,S,V] = svd(E_komega);%svd of the matrix
    d = U(:,1);%update dict
    c_omega = S(1,1)*V(:,1)';
    for i = 1:support_size(2)%update the coefficients
        c(support(i)) = c_omega(i);
    end
end
end
