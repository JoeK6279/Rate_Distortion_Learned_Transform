% 2024-08-27

clear
% close all
clc


N = 8;

% Dataset
load('X_8x8_clic.mat'); % X_test
X_test = reshape(X,[N*N, size(X,3)]);
disp(numel(X_test));
disp(size(X_test));
% disp(size(X));

names = [
"univ_8x8",
"KLT_8x8",
"SOT_8x8",
];

for n_idx=1:length(names)
    name = names(n_idx);
    load(strcat(strcat('final/', name), '.mat'));
    if contains(name, 'KLT')
        M(:,:,1) = V;
    elseif contains(name, 'SOT')
        M(:,:,1) = E;
    else
        M(:,:,1) = high; 
    end

    % Quantization
    Qstep = [20 30 40 50 60]; % Can be changed
    L_q = length(Qstep);

    T_dct = zeros(size(X_test));
    Tq_dct = zeros([size(X_test), length(Qstep)]);
    Tdeq_dct = zeros([size(X_test), length(Qstep)]);
    dct_rec_univ = zeros([size(X_test),length(Qstep)]);
    % 
    Tq_univ = zeros([size(X_test),size(M,3)]);
    rec_univ = zeros([size(X_test),size(M,3)]);

    % Outputs
    my_psnr_univ = zeros(1,length(Qstep));
    my_psnr_dct = zeros(1,length(Qstep));

    

    for ii = 1:size(X_test,2)
        b = X_test(:,ii);
        B = reshape(b,[N,N]);

        %%%%%%%%%%%%% DCT
        T_dct(:,ii) = reshape(dct2(B),[N^2,1]);

        for q = 1:L_q
            Tq_dct(:,ii,q) = round(T_dct(:,ii)/Qstep(q));
            Tdeq_dct(:,ii,q) = Tq_dct(:,ii,q)*Qstep(q);
            dct_rec_univ(:,ii,q) = reshape(idct2(reshape(Tdeq_dct(:,ii,q),[N,N])), [N^2,1]);
        end


        %%%%%%%%%%%%% Univ.
        if contains(name, '_8_')
            for q = 1:size(M,3)
                T_univ = M(:,:,q)*B*M(:,:,q)';
                Tq_univ(:,ii,q) = reshape(round(T_univ/Qstep(q)),[N^2,1]);
                rec_univ(:,ii,q) = reshape(M(:,:,q)\(Qstep(q)*reshape(Tq_univ(:,ii,q),[N,N]))/M(:,:,q)',[N^2,1]);
            end
        else
            for q = 1:L_q
                T_univ = b'*M(:,:,1);
                Tq_univ(:,ii,q) = round(T_univ/Qstep(q));
                % rec_univ(:,ii,q) = Qstep(q)*Tq_univ(:,ii,q)'/M(:,:,q);
                rec_univ(:,ii,q) = Qstep(q)*Tq_univ(:,ii,q)'*M(:,:,1)';
            end
        end



    end

    ent_coef_dct = zeros(size(M,3),size(X_test,1));
    for q = 1:length(Qstep)
        disp(q)
        temp = Tdeq_dct(:,:,q);
        my_mse = sum(sum((T_dct(:)-temp(:)).^2))/numel(X_test);
        % temp  = dct_rec_univ(:,:,q);
        % my_mse = sum(sum((X_test(:)-temp(:)).^2))/numel(X_test);
        my_psnr_dct(q) = 20*log10(255/sqrt(my_mse));
        for k = 1:size(ent_coef_dct,2)
            ent_coef_dct(q,k) = length(my_arith_enco(Tq_dct(k,:,q)))/size(X_test,2);
        end
    end
    my_ent_dct = mean(ent_coef_dct,2); 

    ent_coef_univ = zeros(length(Qstep),size(X_test,1));
    for q = 1:length(Qstep)
        disp(q)
        temp = rec_univ(:,:,q);
        my_mse = sum(sum((X_test(:)-temp(:)).^2))/numel(X_test);
        my_psnr_univ(q) = 20*log10(255/sqrt(my_mse));
        for k = 1:size(ent_coef_univ,2)
            ent_coef_univ(q,k) = length(my_arith_enco(Tq_univ(k,:,q)))/size(X_test,2);
        end
    end
    my_ent_univ = mean(ent_coef_univ,2); 



    figure;
    hold on;
    grid on;
    % plot(total_dct_ent, total_dct_psnr,'-o','MarkerFaceColor','blue');
    plot(my_ent_dct, my_psnr_dct,'-o','MarkerFaceColor','blue');
    % plot(total_univ_ent, total_univ_psnr,'-o','MarkerFaceColor', 'red');
    plot(my_ent_univ, my_psnr_univ,'-o','MarkerFaceColor', 'red');
    legend('DCT', 'univ')
    title(strcat(name, ''))
    saveas(gcf,strcat(name, '_clic.png'));
    savefig(strcat(name, '_clic.fig'));
end





% % 2024-08-27
% 
% clear
% % close all
% clc
% 
% 
% N = 8;
% 
% % Dataset
% load('X_test_8x8.mat'); % X_test
% 
% % Univ. transform
% % "univ_64_factorized_follow_l0_log", "univ_64_factorized_follow_l0_tahn",
% names = [
% %     "univ_64_factorized_follow_diffLR_T",
% % "univ_64_factorized_follow_overfit_T",
% % "univ_64_factorized_follow_nodctinit_T",
% "univ_64_factorized_follow_transpose_T",
% % "univ_64_factorized_follow_lambda_T",
% % "univ_64_factorized_follow_sgd_T",
% % "univ_64_factorized_follow_hardnorm_T",
% % "univ_64_factorized_follow_softnorm_T",
% % "univ_64_factorized_follow_T",
% % "univ_64_factorized_follow_tuning_T"
% ];
% % names = ["univ_64_factorized_residule_traintest_split", "Copy of univ_64_factorized_follow"];
% for n_idx=1:length(names)
%     name = names(n_idx);
%     load(strcat(strcat('new/', name), '.mat'));
%     M(:,:,1) = high; 
%     % M(:,:,1) = M(:,:,1)./sqrt(sum(M(:,:,1).^2));
%     M(:,:,2) = mid;  
%     % M(:,:,2) = M(:,:,2)./sqrt(sum(M(:,:,2).^2));
%     M(:,:,3) = low;  
%     % M(:,:,3) = M(:,:,3)./sqrt(sum(M(:,:,3).^2));
% 
%     % Quantization
%     Qstep = [30 40 50]; % Can be changed
%     L_q = length(Qstep);
% 
%     % T_dct = zeros(size(X_test));
%     % Tq_dct = zeros([size(X_test), length(Qstep)]);
%     % Tdeq_dct = zeros([size(X_test), length(Qstep)]);
%     % dct_rec_univ = zeros([size(X_test),length(Qstep)]);
%     % 
%     Tq_univ = zeros([size(X_test),size(M,3)]);
%     rec_univ = zeros([size(X_test),size(M,3)]);
% 
%     % Outputs
%     my_psnr_univ = zeros(1,length(Qstep));
%     % my_psnr_dct = zeros(1,length(Qstep));
%     for ii = 1:size(X_test,2)
%         b = X_test(:,ii);
%         B = reshape(b,[N,N]);
% 
%         %%%%%%%%%%%%% DCT
%         % T_dct(:,ii) = reshape(dct2(B),[N^2,1]);
%         % 
%         % for q = 1:L_q
%         %     Tq_dct(:,ii,q) = round(T_dct(:,ii)/Qstep(q));
%         %     Tdeq_dct(:,ii,q) = Tq_dct(:,ii,q)*Qstep(q);
%         %     dct_rec_univ(:,ii,q) = reshape(idct2(reshape(Tdeq_dct(:,ii,q),[N,N])), [N^2,1]);
%         % end
%         % 
% 
%         %%%%%%%%%%%%% Univ.
%         if contains(name, '_8_')
%             for q = 1:size(M,3)
%                 T_univ = M(:,:,q)*B*M(:,:,q)';
%                 Tq_univ(:,ii,q) = reshape(round(T_univ/Qstep(q)),[N^2,1]);
%                 rec_univ(:,ii,q) = reshape(M(:,:,q)\(Qstep(q)*reshape(Tq_univ(:,ii,q),[N,N]))/M(:,:,q)',[N^2,1]);
%             end
%         else
%             for q = 1:size(M,3)
%                 T_univ = b'*M(:,:,q);
%                 Tq_univ(:,ii,q) = round(T_univ/Qstep(q));
%                 % rec_univ(:,ii,q) = Qstep(q)*Tq_univ(:,ii,q)'/M(:,:,q);
%                 rec_univ(:,ii,q) = Qstep(q)*Tq_univ(:,ii,q)'*M(:,:,q)';
%             end
%         end
% 
% 
% 
%     end
% 
%     % ent_coef_dct = zeros(size(M,3),size(X_test,1));
%     % for q = 1:length(Qstep)
%     %     disp(q)
%     %     % temp = Tdeq_dct(:,:,q);
%     %     % my_mse = sum(sum((T_dct(:)-temp(:)).^2))/numel(X_test);
%     %     temp  = dct_rec_univ(:,:,q);
%     %     my_mse = sum(sum((X_test(:)-temp(:)).^2))/numel(X_test);
%     %     my_psnr_dct(q) = 20*log10(255/sqrt(my_mse));
%     %     for k = 1:size(ent_coef_dct,2)
%     %         ent_coef_dct(q,k) = length(my_arith_enco(Tq_dct(k,:,q)))/size(X_test,2);
%     %     end
%     % end
%     % my_ent_dct = mean(ent_coef_dct,2); 
% 
%     ent_coef_univ = zeros(size(M,3),size(X_test,1));
%     for q = 1:size(M,3)
%         disp(q)
%         temp = rec_univ(:,:,q);
%         my_mse = sum(sum((X_test(:)-temp(:)).^2))/numel(X_test);
%         my_psnr_univ(q) = 20*log10(255/sqrt(my_mse));
%         for k = 1:size(ent_coef_univ,2)
%             ent_coef_univ(q,k) = length(my_arith_enco(Tq_univ(k,:,q)))/size(X_test,2);
%         end
%     end
%     my_ent_univ = mean(ent_coef_univ,2); 
% 
% 
% 
%     my_ent_dct = [0.523641169347326;0.398997857542073;0.311377344283674];  %reg 30 40 50
%     my_psnr_dct = [34.0057901152669,	32.5471018440808,	31.3810593541616]; %reg 30 40 50
% 
% 
%     figure;
%     hold on;
%     % plot(total_dct_ent, total_dct_psnr,'-o','MarkerFaceColor','blue');
%     plot(my_ent_dct, my_psnr_dct,'-o','MarkerFaceColor','blue');
%     % plot(total_univ_ent, total_univ_psnr,'-o','MarkerFaceColor', 'red');
%     plot(my_ent_univ, my_psnr_univ,'-o','MarkerFaceColor', 'red');
%     legend('DCT', 'univ')
%     title(strcat(name, '_unnorm'))
%     saveas(gcf,strcat(name, '_unnorm.png'));
%     savefig(strcat(name, '_unnorm.fig'));
% end
