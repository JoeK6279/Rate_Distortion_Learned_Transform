
flist = [
"univ_8x8_clic",
"KLT_8x8_clic",
"SOT_8x8_clic",
"univ_16x16_clic",
"KLT_16x16_clic",
"SOT_16x16_clic",
"univ_32x32_clic",
"KLT_32x32_clic",
"SOT_32x32_clic",
];

for i=1:length(flist)
    fig = open(strcat(flist(i),'.fig'));
    h = findobj(gca,'Type','line');
    x=get(h,'Xdata') ;
    y=get(h,'Ydata') ;
    % disp(x);
    % disp(y);
    tmp_x = cell2mat(x);
    tmp_y = cell2mat(y);
    % s=sprintf('%s=RD_Curve([[', flist(i));
    % fprintf(s);
    % s=sprintf('%.5f, ', tmp_x(1,1:5));
    % fprintf(s);
    % fprintf('],[');
    % s=sprintf('%.5f, ', tmp_y(1,1:5));
    % fprintf(s);
    % fprintf(']])\n');
    s=sprintf('%s=RD_Curve([[', flist(i));
    fprintf(s);
    s=sprintf('%.5f, ', tmp_x(2,1:5));
    fprintf(s);
    fprintf('],[');
    s=sprintf('%.5f, ', tmp_y(2,1:5));
    fprintf(s);
    fprintf(']])\n');
    % s=sprintf('%.5f, ', tmp_x(2,:));
    % disp(s);
    % s=sprintf('%.5f, ', tmp_y(2,:));
    % disp(s);
end
