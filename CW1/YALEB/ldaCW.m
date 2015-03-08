function W = ldaCW(data,labels)
[N,F] = size(data);
C = length(unique(labels));
% transpose data and labels to bring to desired form
X = data';
cs = labels';
% order X by class
[cs,inx] = sort(cs);
X = X(:,inx);
% calculate M (class proportions) 
M = bcScatter(X,cs);
Xw = X*(eye(N,N)-M);
[V,D] = eig(Xw'*Xw);
[D,vdx] = sort(-1*diag(D));
D = undiag(-D(1:N-C));
V = V(:,vdx);
V = V(:,1:N-C);
U = Xw*V*(D^-1);
Xb = U'*X*M;
Q = pcaCW(Xb',C-1);
W = U*Q;
end


function [M] = bcScatter(X,cs)
[~,N] = size(X);
M = zeros(N,N);
off = 0;
for c = unique(cs)
    n = length(find(cs==c));
    M(off+1:off+n,off+1:off+n) = ones(n,n)*1/n;
    off = off + n;
end
end


function [D] = undiag(vec)
n = length(vec);
D = zeros(n);
D(cumsum([1,repmat(n+1,1,n-1)])) = vec;
end