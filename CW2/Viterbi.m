function [ P ] = Viterbi(X,A,E,pi)
%VITERBI Performcs Viterbi decoding given the parameters and an observation
%INPUT ARGUMENTS:
%   X: One observation (X = {x1,..xT}
%   A: Transition Probability Matrix
%   E: Emission Probability Matrix
%   pi: probability of choosing a die
%OUTPUT:
%   P: Most probable path
% 
% Reference: 
%   As interpreted from:
%       http://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode    

states = [1,2];

T = length(X);
K = length(states);

% Initialize the two tables 
% T1 stores the probability of the deduced most probable graph
T1 = zeros(K,T); 
% T2 stores the most probable path so far.
T2 = zeros(K,T);

for s = states
    T1(s,1) = pi(s) *E(s,X(s));
end

% Estimate the most probable path
for t = 2:T
    for s = states
        ks = zeros(1,K);
        % estimate probabilities of all possible next steps
        for k = 1:2
            ks(k) = T1(k,t-1) * A(k,s) * E(s,X(t));
        end        
        % most probable next state (along with the probability)
        %[max,argmax]
        [T1(s,t),T2(s,t)] = max(ks); 
    end
end
% decode the path backwards (starting form t=T)
[~,P(T)] = max(T1(:,T)); % argmax_state
for t = T:-1:2       
    P(t-1) = T2(P(t),t);
end
end


