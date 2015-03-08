function [ params ] = emHMMCont(X,pi,A,S,mu)
%emHMMCont Performs the EM algorithm and estimates the parameters of the model
%INPUT
%   X:  set of observations
%   pi: probability to pick the die
%   A: transition matrix
%   S: Gaussian variances
%   mu: Gaussian means
%OUTPUT
%    params.pi = pi \
%    params.A = A    | 
%    params.S = S    | 
%    params.mu = mu  | new parameter estimates
%    params.pi       |
%    params.A        | 
%    params.B       /
%
% Based on the murphy's book and various implementations found online

[N,T] = size(X);

% number of gaussians
M = length(mu);

params.pi = pi;
params.A = A;

for iteration = 1:100    
    iteration
    pi_new = 0 .* pi;
    mu_new = 0 .* mu;    
    A_new = 0 .* A;
    S_new = 0 .* S;  
    Gamma = zeros(M,1);
    for i = 1:N    
        % calculate emission prob
        B = zeros(M,T);
        for g = 1:M
            B(g,:) = mvnpdf(X(i,:)',mu(:,g),S(:,g));
        end
        
        % compute forward probabilities
        C1 = zeros(M,T);
        scale = zeros(1,T);
        
        C1(:,1) = pi .* B(:,1);
        scale(1) = 1/sum(C1(:,1));
        
        for t = 2:T
            C1(:,t) = A'* C1(:,t-1).*B(:,t);
            scale(t) = 1./sum(C1(:,t));
            C1(:,t) = C1(:,t) * scale(t);
        end
        %compute backward probabilities
        C2 = zeros(M,T);
        C2(:,T) = ones(M,1);
        for t = T-1:-1:1
            C2(:,t) = A*(B(:,t+1).*C2(:,t+1));
            C2(:,t) = C2(:,t) * scale(t);
        end
        % E-step - calculate expectations
        gamma = C1.*C2;
        gamma = bsxfun(@times,gamma,1./sum(gamma));
        
        Xi = A .* 0;
        for t = 1:T-1
            xi = (C1(:,t) * (C2(:,t+1) .* B(:,t+1))' ).* A;
            Xi = Xi + xi / sum(sum(xi));
        end
        
       % M-step recalculate parameters to maximise        
        pi_new = pi_new + gamma(:,1);
        A_new = A_new + Xi;
        mu_new = mu_new + X(i,:)*gamma';
        for g = 1:M
            S_new(:,g) = S_new(:,g) + bsxfun(@times,X(i,:),gamma(g,:))* X(i,:)';
        end
        Gamma = Gamma + sum(gamma,2);
    end
    
    pi = pi_new/N;
    A = bsxfun(@times,A_new,1./sum(A_new,2));
    mu = bsxfun(@times,mu_new,1./Gamma');
    
    S = bsxfun(@times,S_new,1./reshape(Gamma,1,M));
    for g = 1:M
        S(:,g) = S(:,g) - mu(:,g)*mu(:,g)';
        S(:,g) = (S(:,g) + S(:,g)') / 2;
    end
    
    
    if  isequal(params.pi,pi) && isequal(params.A,A)  &&  isequal(params.S,S) && isequal(params.mu,mu)
        return
    end

    params.pi = pi;
    params.A = A;
    params.S = S;
    params.mu = mu; 
end