function [T,Tdeq,bit_MTS] = chooseBestMTS(B,Qstep,lambda)

numb_tr = 5;
my_mse = zeros(1,numb_tr);
my_rate = zeros(1,numb_tr);
V = zeros(size(B,1),size(B,2),numb_tr);
Vdeq = zeros(size(B,1),size(B,2),numb_tr);

T = dct2(B); Tdeq = Qstep*round(T/Qstep);
V(:,:,1) = T; Vdeq(:,:,1) = Tdeq;
my_mse(1) = immse(T,Tdeq);
my_rate(1) = my_entropy(zeros(1,2),Tdeq(:));

T = MTS_00(B); Tdeq = Qstep*round(T/Qstep);
V(:,:,2) = T; Vdeq(:,:,2) = Tdeq;
my_mse(2) = immse(T,Tdeq);
my_rate(2) = my_entropy(zeros(1,2),Tdeq(:));

T = MTS_01(B); Tdeq = Qstep*round(T/Qstep);
V(:,:,3) = T; Vdeq(:,:,3) = Tdeq;
my_mse(3) = immse(T,Tdeq);
my_rate(3) = my_entropy(zeros(1,2),Tdeq(:));

T = MTS_10(B); Tdeq = Qstep*round(T/Qstep);
V(:,:,4) = T; Vdeq(:,:,4) = Tdeq;
my_mse(4) = immse(T,Tdeq);
my_rate(4) = my_entropy(zeros(1,2),Tdeq(:));

T = MTS_11(B); Tdeq = Qstep*round(T/Qstep);
V(:,:,5) = T; Vdeq(:,:,5) = Tdeq;
my_mse(5) = immse(T,Tdeq);
my_rate(5) = my_entropy(zeros(1,2),Tdeq(:));

%%%%%%%%%%%%%%%%% Start of RDO %%%%%%%%%%%%%%%%%
[~,idx_rd] = min(my_mse+lambda*my_rate);
T = V(:,:,idx_rd); Tdeq = Vdeq(:,:,idx_rd);
%%%%%%%%%%%%%%%%% End of RDO %%%%%%%%%%%%%%%%%
if idx_rd == 1
    bit_MTS = 1;
else
    bit_MTS = 3;
end