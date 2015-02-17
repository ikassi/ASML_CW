function [ W ] = fasticaCW(X,C)
[M,N] = size(X);
% sphearing
S = cov(X');
[E,D] = eig(S);
X = (E*real(D^(-1/2))*E')*X;
X = X';

% implemented as interpreted from the description on wikipedia
% http://en.wikipedia.org/wiki/FastICA
W = zeros(C,N);

% iterate over all dimentions
for p = 1:C
    p
    % initialize wp randomly
    wp = rand(N,1);
    wp_diff = true;
    % repeat until wp converges
    while wp_diff
        % calculate newton updates
        wp = (1/M)*X*g(wp'*X) - (1/M)*g2(wp'*X)*ones(M,1)*wp;
        s = zeros(N,1);
        % we subtract from the projections of the previously estimated vectors
        for j=1:(p-1)
            s = s + wp'*W(j,:)'*W(j,:)';
        end
        wp = wp - s;
        % renormalize
        wp = wp/norm(wp);
        % check for convergence
        wp_diff = wp ~= W(p,:)';
        W(p,:) = wp';
    end
end
W=W';
end

%first derivative
function du = g(u)
du = tanh(u)';
end
%second derivative
function d2u = g2(u)
d2u = (1-tanh(u).^2);
end

