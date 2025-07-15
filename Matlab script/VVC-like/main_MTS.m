% clear
% % close all
% clc

function [dct_psnr,dct_bpp] = main_MTS(N, X)

% Dataset
% N = 8;
% temp = load('X_large_N=16.mat'); % X_test
% temp = load('../../X_32x32_imagenet.mat'); % X_test

% disp(size(temp.X));

% X = reshape(temp.X(:,:,floor(size(temp.X,3)/5)*4:size(temp.X,3)),[N*N, size(temp.X,3)-floor(size(temp.X,3)/5)*4+1]);
% X = reshape(temp.X(:,:,floor(size(temp.X,3)/20)*19:size(temp.X,3)),[N*N, size(temp.X,3)-floor(size(temp.X,3)/20)*19+1]);
% X = reshape(temp.X(:,:,floor(199453/20)*19:199453),[N*N, 199453-floor(199453/20)*19+1]);


% temp = load('VideoDatasetBS8.mat'); % X_test
% X=[];
% for t=1:35
%     tmp = cell2mat(temp.I_cols(1,t));
%     X = [X,tmp(:,fix(size(tmp,2)/6)*5:size(tmp,2))];
% end


disp(numel(X));
disp(size(X));

% Quantization
% QP = 25:5:45;
% Qstep = (2^(1/6)).^(QP-4);
Qstep = 20:10:60;
QP = log2(Qstep.^6)+4;
L_q = length(QP);
disp(QP);

% Outputs
my_psnr_learned = zeros(1,length(QP));
my_bpp_learned_coeff = zeros(1,length(QP));
my_bpp_learned_index = zeros(1,length(QP));
my_psnr_mts = zeros(1,length(QP));
my_bpp_mts_coeff = zeros(1,length(QP));
my_bpp_mts_index = zeros(1,length(QP));

for q = 1:L_q % Quantization step iteration

    fprintf('Quantizzazione %d di %d\n',q,L_q)

    lambda = 0.57*2.^((QP(q)-12)/3);

    % MTS
    T_mts = zeros(N,N,size(X,2));
    T_mts_deq = zeros(N,N,size(X,2));
    bit_MTS = zeros(1,size(X,2));
    for ii = 1:size(X,2)
        B = reshape(X(:,ii),[N,N]);
        [T_mts(:,:,ii),T_mts_deq(:,:,ii),g] = chooseBestMTS(B,Qstep(q),lambda);
        bit_MTS(ii) = g;
    end

    my_psnr_mts(q) = 20*log10(255/sqrt(sum((T_mts(:)-T_mts_deq(:)).^2)/numel(X)));

    ent_coef = zeros(1,size(X,1));
    T_mts_deq_w = reshape(T_mts_deq,[N^2, size(X,2)]);
    for k = 1:length(ent_coef)
        ent_coef(k) = length(my_arith_enco(T_mts_deq_w(k,:)/Qstep(q)))/size(X,2);
    end
    % code_MTS = my_arith_enco(T_mts_deq/Qstep(q));
    my_bpp_mts_coeff(q) = mean(ent_coef);
    my_bpp_mts_index(q) = sum(bit_MTS)/numel(X);
end
dct_bpp = my_bpp_mts_coeff + my_bpp_mts_index;
dct_psnr= my_psnr_mts;
% s = sprintf('64 = RD_Curve([[');
% fprintf(s);
% s=sprintf('%.5f, ', my_bpp_mts_coeff(1,1:5) + my_bpp_mts_index(1,1:5));
% fprintf(s);
% fprintf('],[');
% s=sprintf('%.5f, ', my_psnr_mts(1,1:5));
% fprintf(s);
% fprintf(']])\n');