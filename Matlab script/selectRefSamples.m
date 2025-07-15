function [refAbove, refLeft] = selectRefSamples(img_dec, r, c, N)

% selectRefSamples Selects the reference samples extracting the appropriate
% decoded neighboring pixels.
%
% [refAbove, refLeft] = selectRefSamples(img_dec, r, c) operates as
% described in "Intra Coding of the HEVC Standard".
% See https://ieeexplore.ieee.org/document/6317153

refAbove = zeros(1,2*N+1);
refLeft = zeros(1,2*N+1);

% refAbove
if r == 1 % First blocks row, then refAbove -> 128
    refAbove = refAbove + 128; 
elseif c == size(img_dec,2)/N % There are not top-right samples, then the last value is copied
    refAbove(1:N+1) = img_dec((r-1)*N, (c-1)*N:c*N);
    refAbove(N+2:end) = refAbove(N+1);
else % Caso standard
    if c == 1
        refAbove(1) = 128;
        refAbove(2:end) = img_dec((r-1)*N, (c-1)*N+1:N*(c+1));
    else
        refAbove = img_dec((r-1)*N, (c-1)*N:N*(c+1));
    end
    
end
% refLeft
if c == 1 % A sinistra non ho niente, refLeft -> 128
    refLeft = refLeft + 128; 
else % Non ho i campioni in basso a sinistra, allora copio l'ultimo valore (il caso standard non c'? mai a causa dell'ordine dello scanning dei blocchi)
    refLeft(1) = refAbove(1);
    refLeft(2:N+1) = img_dec((r-1)*N+1:r*N, (c-1)*N);
    refLeft(N+2:end) = refLeft(N+1);
end