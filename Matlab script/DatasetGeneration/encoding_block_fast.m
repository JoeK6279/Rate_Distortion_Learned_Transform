function [B_res,pm] = encoding_block_fast(img, r, c, N)

% Number of prediction modes: from 0 to 34
numbPredModes = 35;

%%%%%%%%%%%%%%%%%%%%% Reference samples %%%%%%%%%%%%%%%%%%%%%
[refAbove, refLeft] = selectRefSamples(img, r, c, N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% Intra-prediction %%%%%%%%%%%%%%%%%%%%%
B = img((r-1)*N+1:r*N, (c-1)*N+1:c*N);

my_mse = zeros(1, length(numbPredModes));
for dirMode = 0:numbPredModes-1

    % Intra-prediction computation
    B_pred = intraPrediction(dirMode, refAbove, refLeft, N); % Prediction block
    B_res = B-B_pred; % Residual block

    %%%%%%%%%%%%%%%%%%%%%%%%%%%

    t = dct2(B_res);
    t_q = round(t); % Quantized coefficients
    t_deq = t_q; % Dequantized coefficients
    if isequal(t_q(2:end), zeros(1,N^2-1))
        my_mse(dirMode+1) = NaN;
    else
        B_dec = round(idct2(t_deq) + B_pred); % Decoded block
        my_mse(dirMode+1) = sum(sum((B_dec-B).^2));
    end

end

[m,iM] = min(my_mse);
pm = iM-1;
if isfinite(m)
    B_pred = intraPrediction(pm, refAbove, refLeft, N); 
    B_res = B-B_pred; 
else
    B_res = [];
end
