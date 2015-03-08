function [ params ] = emHMM(X,pi,A,B)
%emHMM Performs the EM algorithm and estimates the parameters of the model
%INPUT
%   X:  set of observations
%   pi: probability to pick the die
%   B: Emmision Probability Matrix
%   A: transition matrix
%OUTPUT
%    params.pi \
%    params.A   | new parameter estimates
%    params.B  /
%
% Based on the algorithm description from Murphy's book

% N: number of sequences
% T: length of sequences
% J: length of alphabet
% K: number of hidden variables(dices)
[N,T] = size(X);
[K,J] = size(B);
E = B';

params.pi = pi;
params.A = A;
params.B = B;

for iteration = 1:1000    
    % estimate holders
    pi_new = 0 .* pi;
    A_new = 0 .* A;
    B_new = 0 .* B;
    
    for i = 1:N    
        % For every squence compute forward and backward probabilities    
        % Forward - alpha    
        C1 = pi .* E(X(i,1),:)';    
        scale(1) = 1/sum(C1(:));
        for t = 2:T
            C1 = [C1,E(X(i,t),:)' .* (A' * C1(:,t-1))];
            scale(t) = 1./sum(C1(:));
            C1(:,1) = C1(:,1) * scale(t);           
        end            
        % Backward - beta
        C2 = ones(K,1);
        for t = T-1:-1:1      
            C2 = [scale(t)*A*(C2(:,1).* E(X(i,t+1),:)'),C2];                
        end           
        
        % E-step - calculate expectations
        
        gamma = C1.*C2;
        gamma = bsxfun(@times,gamma,1./sum(gamma));        
       
        Xi = A .* 0;
        for t = 1:T-1
            xi = (C1(:,t) * (C2(:,t+1) .* B(:,X(i,t+1)))' ).* A;
            Xi = Xi + xi / sum(sum(xi));
        end
        % M-step recalculate parameters to maximise
        pi_new = pi_new + gamma(:,1);
        A_new = A_new + Xi;        
        
        for j = 1:J
            B_new(:,j) = B_new(:,j) + sum(gamma(:,X(i,:)==j),2);
        end        
    end
    
    % replace old estimates
    pi = pi_new / N;
    A = bsxfun(@times,A_new,1./sum(A_new,2));
    B = bsxfun(@times,B_new,1./sum(B_new,2));
    
    % check if it converges - stops changing
    if  isequal(params.pi,pi) && isequal(params.A,A)  &&  isequal(params.B,B)
        return
    end    
    
    params.pi = pi;
    params.A = A;
    params.B = B;    
end
end


% 
% function P = xjoint(X,B,pi,l)
% [~,T] = size(X);
% [~,K] = size(B);
% P = 1;
% for t = 1:T
%     state = X(l,t);   
%     c = 0;
%     for k = 1:K
%         c = c + pi(k) * B(state,k);
%     end
%     P = P*c;
% end
% end
