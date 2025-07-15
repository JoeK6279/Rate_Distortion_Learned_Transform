clear;
clc;

Qstep = 20:10:60;
dct_psnr = zeros(1,length(Qstep));
dct_bpp = zeros(1,length(Qstep));

temp = load('VideoDatasetBS8.mat'); % X_test
X=[];
for t=1:35
    tmp = cell2mat(temp.I_cols(1,t));
    X = [X,tmp(:,fix(size(tmp,2)/6)*5:size(tmp,2))];
end

load('../0923/univ_8x8_newtraining.mat')
M = double(high);


% [dct_psnr,dct_bpp] = main_MTS(8, X);
[ours_psnr,my_bpp_learned_coeff,my_bpp_learned_index] = main_MTS_with_learned(8, X, M);
% s = sprintf('VVC_DCT_64 = RD_Curve([[');
% fprintf(s);
% s=sprintf('%.5f, ', dct_bpp);
% fprintf(s);
% fprintf('],[');
% s=sprintf('%.5f, ', dct_psnr);
% fprintf(s);
fprintf(']])\n');
s = sprintf('VVC_learned_64_dctfor01 = RD_Curve([[');
fprintf(s);
s=sprintf('%.5f, ', my_bpp_learned_coeff+my_bpp_learned_index);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', ours_psnr);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', my_bpp_learned_coeff);
fprintf(s);
fprintf(']])\n');


temp = load('X_large_N=16.mat'); % X_test
X = reshape(temp.X(:,:,floor(size(temp.X,3)/5)*4:size(temp.X,3)),[16*16, size(temp.X,3)-floor(size(temp.X,3)/5)*4+1]);


load('../0923/univ_16x16_newtraining.mat')
M = double(high);


% [dct_psnr,dct_bpp] = main_MTS(16, X);
[ours_psnr,my_bpp_learned_coeff,my_bpp_learned_index] = main_MTS_with_learned(16, X, M);
% s = sprintf('VVC_DCT_256 = RD_Curve([[');
% fprintf(s);
% s=sprintf('%.5f, ', dct_bpp);
% fprintf(s);
% fprintf('],[');
% s=sprintf('%.5f, ', dct_psnr);
% fprintf(s);
% fprintf(']])\n');
s = sprintf('VVC_learned_256_dctfor01 = RD_Curve([[');
fprintf(s);
s=sprintf('%.5f, ', my_bpp_learned_coeff+my_bpp_learned_index);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', ours_psnr);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', my_bpp_learned_coeff);
fprintf(s);
fprintf(']])\n');

temp = load('../../X_32x32_imagenet.mat'); % X_test
X = reshape(temp.X(:,:,floor(size(temp.X,3)/20)*19:size(temp.X,3)),[32*32, size(temp.X,3)-floor(size(temp.X,3)/20)*19+1]);

load('../0923/univ_32x32x_newtraining.mat')
M = double(high);
% [dct_psnr,dct_bpp] = main_MTS(32, X);
[ours_psnr,my_bpp_learned_coeff,my_bpp_learned_index] = main_MTS_with_learned(32, X, M);
% s = sprintf('VVC_DCT_1024 = RD_Curve([[');
% fprintf(s);
% s=sprintf('%.5f, ', dct_bpp);
% fprintf(s);
% fprintf('],[');
% s=sprintf('%.5f, ', dct_psnr);
% fprintf(s);
% fprintf(']])\n');
s = sprintf('VVC_learned_1024_dctfor01 = RD_Curve([[');
fprintf(s);
s=sprintf('%.5f, ', my_bpp_learned_coeff+my_bpp_learned_index);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', ours_psnr);
fprintf(s);
fprintf('],[');
s=sprintf('%.5f, ', my_bpp_learned_coeff);
fprintf(s);
fprintf(']])\n');