function KSVD
a = imread('cameraman_noisy.png'); %read the noisy image 
image_data = double(a); %convert the data into double
x = x_matrix(image_data); %calls function x_matrix that takes 8x8 patches of the image and converts them into 64x1
dict = randn(64,256); %generate a dictionary using random numbers
for k = 1:256 %normalise each column
    dict(:,k) = dict(:,k)/norm(dict(:,k));
end
for j = 1:100 % number of iterations for the dictionary learning 
    c = OMP(3, x, dict); %sparse coding
    [dict, c] = dict_learning(x,c,dict); %dictionary learning
    disp(j)
end
x_final = dict*c; %build the final image
b = uint8(final_matrix(x_final));%convert the data back to integer values
imwrite(b, 'draft3_100.png') %write the image into a file
subplot(1,2,1); imshow(b); %show the new image produced
subplot(1,2,2); imshow(a); %show the original noisy image
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

function [d, c] = dict_learning(x,c,d)
for i = 1:256
    E_k = x;
    idx = setdiff([1:256],i);
    for j = idx
        E_k = E_k - d(:,j)*c(j,:);
    end
    [d(:,i), c(i,:)] = dict_col(E_k, d(:,i), c(i,:));
end
end

function [d, c] = dict_col(E_k, d, c)
support = find(c);
if isempty(support) ~= 1
    support_size = size(support);
    c_omega = zeros(support_size(2),1);
    E_komega = zeros(64,support_size(2));
    for i = 1:support_size(2)
        c_omega(i,1) = c(support(i));
        E_komega(:,i) = E_k(:,support(i));
    end
    [U,S,V] = svd(E_komega);
    d = U(:,1);
    c_omega = S(1,1)*V(:,1)';
    for i = 1:support_size(2)
        c(support(i)) = c_omega(i);
    end
end
end

% Reshapes the 64X1024 matrix into a 256X256 matrix
function f = final_matrix(c)
f = zeros(256,256);
count = 1;
for i = 0:31
    for j = 0:31
        f(1+i*8:8+i*8,1+j*8:8+j*8) = reshape(c(:,count),[8,8]);
        count = count + 1;
    end
end
end
