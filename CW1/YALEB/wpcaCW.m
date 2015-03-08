function [U] = wpcaCW(data,d)
[N,F] = size(data);
%centering the data
m = mean(data,1);
X = data - repmat(m,N,1);
% eigen-analysis of the identity (X'X and XX' have same eigenvalues)
S = X*X';
[evc,evl] = eig(S);
% calculate the eigenvectors of the covariance matrix
evc = X'*evc*evl^(-1/2);
% sorting with regards to the eigenvalues
evl = diag(evl);
[evl,ind] = sort(-1*evl);
evl = -1*evl;
U = evc(:,ind)*(undiag(abs(evl))^-0.5);
% returning the dim reduc transofrm for d<=N-1 (d largest eigenvalues)
U = U(:,1:d);
end

function [D] = undiag(vec)
n = length(vec);
D = zeros(n);
D(cumsum([1,repmat(n+1,1,n-1)])) = vec;
end