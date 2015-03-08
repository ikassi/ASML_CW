function [U] = pcaCW(data,d)
[N,~] = size(data);
%centering the data
m = mean(data,1);
X = data - repmat(m,N,1);
% eigen-analysis of the identity (X'X and XX' have same eigenvalues)
S = X*X';
[evc,evl] = eig(S);
% calculate the eigenvectors of the covariance matrix
evc = X'*evc*(abs(evl)^(-1/2));
% sorting with regards to the eigenvalues
evl = diag(evl);
[~,ind] = sort(-1*evl);
U = evc(:,ind);
% returning the dim reduc transofrm for d<=N-1 (d largest eigenvalues)
U = U(:,1:d);
end