function g = denoising(f)
[D, C] = kdict(f, 256, 3);
x_final = D*C; %build the final image
g = uint8(final_matrix(x_final));%convert the data back to integer values
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

