function [W] = lppCW(data,k,sd,d)
X = data';
% construct distance graph S
S = distanceGraph(X,k);
% apply the gaussian basis function
S = gaussianKernel(S,sd);
% compute weight matrix D 
D = diag(sum(S, 2));
% we compute the laplacian
L = D - S;
% make XDX' and XLX' symmetric
L(isnan(L)) = 0;
D(isnan(D)) = 0;
L(isinf(L)) = 0;
D(isinf(D)) = 0;

DP = X * D * X';
LP = X * L * X';
DP = (DP + DP') / 2;
LP = (LP + LP') / 2;

% perform eigenanalysis
[V,L] =  eig(LP, DP);%eig(St);
[~,vdx] = sort(diag(L));
% return ordered components
W = V(:,vdx(1:d));
end



function Sg = gaussianKernel(S,sd)
[~,N] = size(S);
S = S .^ 2;
Sg = S;
for i =1:N
    for j = 1:N
        if S(i,j) ~= 0
            Sg(i,j) = -exp(S(i,j)/(2*sd*sd));
        end
    end
end
end
function S = distanceGraph(X,k)
[~,N] = size(X);
S = zeros(N,N);
for i = 1:N
    for j = i:N
        A = X(:,i);
        B = X(:,j);
        D = norm(A-B);
        S(i,j) = D;
        S(j,i) = D;
    end
    % only keep k nearest
    [~,ddx] = sort(S(i,:));
    ml = zeros(1,N);    
    ml(ddx(1:k+1)) = 1;
    S(i,:) = S(i,:) .* ml;        
end


%make it symmetric
for i = 1:N
    for j = i:N
        d = max(S(i,j),S(j,i));
        if d>0            
            S(j,i) = d;
            S(j,i) = d;
        end
    end
end
end

