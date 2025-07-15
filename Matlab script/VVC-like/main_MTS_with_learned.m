% clear
% % close all
% clc

function [ours_psnr,my_bpp_learned_coeff,my_bpp_learned_index] = main_MTS_with_learned(N, X, M)

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

% load('../0923/univ_8x8_newtraining.mat')
% M = double(high);

% Quantization
Qstep = 20:10:60;
QP = log2(Qstep.^6)+4;
L_q = length(QP);
disp(QP);

% Outputs
my_psnr_learned = zeros(1,length(QP));
my_bpp_learned_coeff = zeros(1,length(QP));
my_bpp_learned_index = zeros(1,length(QP));

for q = 1:L_q % Quantization step iteration

    fprintf('Quantizzazione %d di %d\n',q,L_q)

    lambda = 0.57*2.^((QP(q)-12)/3);

    % LEARNED
    T_learned = zeros(size(X));
    T_learned_deq = zeros(size(X));
    T_learned_rec = zeros(size(X));
    bit_learned = zeros(1,size(X,2));
    for ii = 1:size(X,2)
        b = X(:,ii);
        [T_learned(:,ii),T_learned_deq(:,ii),T_learned_rec(:,ii),g]...
            = chooseBestMTS_learned(b,Qstep(q),lambda,M);
        bit_learned(ii) = g;
    end

    my_psnr_learned(q) = 20*log10(255/sqrt(sum((X(:)-T_learned_rec(:)).^2)/numel(X)));

    ent_coef = zeros(1,size(X,1));
    ent_coef_learned = zeros(1,size(X,1));
    for k = 1:length(ent_coef)
        ent_coef(k) = length(my_arith_enco(T_learned_deq(k,bit_learned==3)/Qstep(q)))/sum(bit_learned==3);
        ent_coef_learned(k) = length(my_arith_enco(T_learned_deq(k,bit_learned==1)/Qstep(q)))/sum(bit_learned==1);
    end
    % code_learned = my_arith_enco(T_learned_deq/Qstep(q));
    my_bpp_learned_coeff(q) = sum(bit_learned==3)*mean(ent_coef)/size(X,2) + sum(bit_learned==1)*mean(ent_coef_learned)/size(X,2);
    my_bpp_learned_index(q) = sum(bit_learned)/numel(X);
end
ours_bpp = my_bpp_learned_coeff+my_bpp_learned_index;
ours_psnr = my_psnr_learned;
% s = sprintf('RD_Curve([[');
% fprintf(s);
% s=sprintf('%.5f, ', my_bpp_mts_coeff(1,1:5) + my_bpp_mts_index(1,1:5));
% fprintf(s);
% fprintf('],[');
% s=sprintf('%.5f, ', my_psnr_learned(1,1:5));
% fprintf(s);
% fprintf(']])\n');