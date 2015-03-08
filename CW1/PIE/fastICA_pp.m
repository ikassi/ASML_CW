function [Xnew] = fastICA_pp(X)
%centering
Xnew = X - repmat(mean(X,1),size(X,1),1);
end

