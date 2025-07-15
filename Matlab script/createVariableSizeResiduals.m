clear
close all
clc

N = 32;
num_blocks = 0; % TO CHANGE: Better to know the third size
X = zeros(N,N,num_blocks); 
PM_vec = zeros(1,num_blocks); 
count = 1;

% Folders and files
fld_images = 'Imageset_Completo/'; % TO CHANGE
imagebase = dir(fullfile(fld_images, '*.tiff')); % TO CHANGE

for i = 1:length(imagebase)

    % Image loading and pre-processing operations
    strg = imagebase(i).name;
    fprintf('Image %s\n', strg)

    img = imread(strcat(fld_images, strg));
    if length(size(img))==3
        img = rgb2gray(img);
    end
    img = double(img);

    for r = 1:size(img,1)/N
        for c = 1:size(img, 2)/N
            [B_res,pm] = encoding_block_fast(img, r, c, N);
            if ~isempty(B_res)
                X(:,:,count) = B_res;
                PM_vec(count) = pm;
                count = count + 1;
            end
        end
    end

end
PM_vec = PM_vec(1:count-1);
X = X(:,:,1:count-1);

save("In/X_nameOfTheFile.mat","X") % TO CHANGE: insert the name
save("In/PM_nameOfTheFile.mat","PM_vec") % TO CHANGE: insert the name