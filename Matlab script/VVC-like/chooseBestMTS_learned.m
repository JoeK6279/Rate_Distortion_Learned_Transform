function [T,Tdeq,rec,bit_MTS] = chooseBestMTS_learned(b,Qstep,lambda,M)

N = sqrt(length(b));
B = reshape(b,[N,N]);

numb_tr = 5;
my_mse = zeros(1,numb_tr);
my_rate = zeros(1,numb_tr);
V = zeros(length(b),numb_tr);
Vdeq = zeros(length(b),numb_tr);
Vrec = zeros(length(b),numb_tr);

T = b'*M; Tdeq = Qstep*round(T/Qstep); rec = Tdeq*M'; rec = rec';
my_mse(1) = immse(b,rec);
my_rate(1) = my_entropy(zeros(1,2),Tdeq(:)); 
V(:,1) = T; Vdeq(:,1) = Tdeq; Vrec(:,1) = rec;

T = MTS_00(B); Tdeq = Qstep*round(T/Qstep); rec = IMTS_00(Tdeq); 
my_mse(2) = immse(T,Tdeq);
my_rate(2) = my_entropy(zeros(1,2),Tdeq(:));
V(:,2) = reshape(T,[N^2,1]); Vdeq(:,2) = reshape(Tdeq,[N^2,1]); Vrec(:,2) = reshape(rec,[N^2,1]);

% T = MTS_01(B); Tdeq = Qstep*round(T/Qstep); rec = IMTS_01(Tdeq);
T = dct2(B); Tdeq = Qstep*round(T/Qstep); rec = idct2(Tdeq);
my_mse(3) = immse(T,Tdeq);
my_rate(3) = my_entropy(zeros(1,2),Tdeq(:));
V(:,3) = reshape(T,[N^2,1]); Vdeq(:,3) = reshape(Tdeq,[N^2,1]); Vrec(:,3) = reshape(rec,[N^2,1]);

T = MTS_10(B); Tdeq = Qstep*round(T/Qstep); rec = IMTS_10(Tdeq); 
my_mse(4) = immse(T,Tdeq);
my_rate(4) = my_entropy(zeros(1,2),Tdeq(:));
V(:,4) = reshape(T,[N^2,1]); Vdeq(:,4) = reshape(Tdeq,[N^2,1]); Vrec(:,4) = reshape(rec,[N^2,1]);

T = MTS_11(B); Tdeq = Qstep*round(T/Qstep); rec = IMTS_11(Tdeq); 
my_mse(5) = immse(T,Tdeq);
my_rate(5) = my_entropy(zeros(1,2),Tdeq(:));
V(:,5) = reshape(T,[N^2,1]); Vdeq(:,5) = reshape(Tdeq,[N^2,1]); Vrec(:,5) = reshape(rec,[N^2,1]);

%%%%%%%%%%%%%%%%% Start of RDO %%%%%%%%%%%%%%%%%
[~,idx_rd] = min(my_mse+lambda*my_rate);
T = V(:,idx_rd); Tdeq = Vdeq(:,idx_rd); rec = Vrec(:,idx_rd);
%%%%%%%%%%%%%%%%% End of RDO %%%%%%%%%%%%%%%%%
if idx_rd == 1
    bit_MTS = 1;
else
    bit_MTS = 3;
end