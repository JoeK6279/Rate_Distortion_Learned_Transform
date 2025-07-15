function [ent, x] = my_entropy(x,y)

% x = seq of coeff and counts
% y = nuovo blocco

[y_un,~,ic] = unique(y);
y_counts = accumarray(ic,1);

for k = 1:length(y_un)
    ind = find(y_un(k)==x(:,1), 1);
    if isempty(ind)
        x(end+1,:) = [y_un(k), y_counts(k)];
    else
        x(ind,2) = x(ind,2) + y_counts(k);
    end
end
p = x(:,2)/sum(x(:,2));
ent = -sum(p.*log2(p));