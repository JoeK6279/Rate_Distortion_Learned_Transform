% 2024-08-27

clear
% close all
clc

N = 8;

% Dataset
load('Input\X_test_8x8.mat'); % X_test

% Univ. transform
load('Input\univ_64_factorized_residule_traintest_split.mat');
M(:,:,1) = high; %M(:,:,1) = M(:,:,1)./sqrt(sum(M(:,:,1).^2));
M(:,:,2) = mid; %M(:,:,2) = M(:,:,2)./sqrt(sum(M(:,:,2).^2));
M(:,:,3) = low; %M(:,:,3) = M(:,:,3)./sqrt(sum(M(:,:,3).^2));

% Quantization
Qstep = [30 40 50]; % Can be changed
L_q = length(Qstep);

T_dct = zeros(size(X_test));
Tq_dct = zeros([size(X_test), length(Qstep)]);
Tdeq_dct = zeros([size(X_test), length(Qstep)]);

Tq_univ = zeros([size(X_test),size(M,3)]);
rec_univ = zeros([size(X_test),size(M,3)]);

% Outputs
my_psnr_univ = zeros(1,length(Qstep));
my_psnr_dct = zeros(1,length(Qstep));
% my_ent_univ = zeros(1,length(Qstep));
% my_ent_dct = zeros(1,length(Qstep));
for ii = 1:size(X_test,2)
    b = X_test(:,ii);
    B = reshape(b,[N,N]);

    %%%%%%%%%%%%% DCT
    T_dct(:,ii) = reshape(dct2(B),[N^2,1]);

    for q = 1:L_q
        Tq_dct(:,ii,q) = round(T_dct(:,ii)/Qstep(q));
        Tdeq_dct(:,ii,q) = Tq_dct(:,ii,q)*Qstep(q);
    end

    %%%%%%%%%%%%% Univ.
    for q = 1:size(M,3)
        T_univ = b'*M(:,:,q)/255;
        Tq_univ(:,ii,q) = round(T_univ);
        rec_univ(:,ii,q) = Tq_univ(:,ii,q)'/M(:,:,q);
    end 
    
end

ent_coef_dct = zeros(size(M,3),size(X_test,1));
for q = 1:length(Qstep)
    disp(q)
    temp = Tdeq_dct(:,:,q);
    my_mse = sum(sum((T_dct(:)-temp(:)).^2))/numel(X_test);
    my_psnr_dct(q) = 20*log10(255/sqrt(my_mse));
    for k = 1:size(ent_coef_dct,2)
        ent_coef_dct(q,k) = length(my_arith_enco(Tq_dct(k,:,q)))/size(X_test,2);
    end
end
my_ent_dct = mean(ent_coef_dct,2); 

ent_coef_univ = zeros(size(M,3),size(X_test,1));
for q = 1:size(M,3)
    disp(q)
    temp = rec_univ(:,:,q)*255;
    my_mse = sum(sum((X_test(:)-temp(:)).^2))/numel(X_test);
    my_psnr_univ(q) = 20*log10(255/sqrt(my_mse));
    for k = 1:size(ent_coef_univ,2)
        ent_coef_univ(q,k) = length(my_arith_enco(Tq_univ(k,:,q)))/size(X_test,2);
    end
end
my_ent_univ = mean(ent_coef_univ,2); 