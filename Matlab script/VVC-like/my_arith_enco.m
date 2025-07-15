function code = my_arith_enco(T)

    % Bring symbols from 1 to length(unique(T))
    T_mod = zeros(size(T));
    un = unique(T);
    for k = 1:length(un)
        T_mod(T==un(k)) = k;
    end
    
    counts = histc(T_mod(:), 1:length(un));
    if isscalar(counts)
        code = [];
    else
        code = arithenco(T_mod(:),counts);
    end
    


% function code = my_arith_enco(T)

% % Bring symbols from 1 to length(unique(T))
% T_mod = zeros(size(T));
% un = unique(T);
% for k = 1:length(un)
%     T_mod(T==un(k)) = k;
% end

% counts = histc(T_mod(:), 1:length(un));
% if numel(counts)==1
%     code = 1;
% else
%     code = length(arithenco(T_mod(:),counts));
% end
