function [B_pred, ver] = intraPrediction(dirMode, refAbove, refLeft, N)

% intraPrediction Returns the prediction block given the prediction mode
% and the reference samples
%
% B_pred = intraPrediction(DIRMODE, REFABOVE, REFLEFT) operates as
% described in "Intra Coding of the HEVC Standard".
% See https://ieeexplore.ieee.org/document/6317153 and 
% https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/ for implementation
% details

VER_IDX = 26;
HOR_IDX = 10;

B_pred = zeros(N);

if dirMode==0
    %%%%%%%%%%%%%%%%%%%% DC MODE %%%%%%%%%%%%%%%%%%%%
    B_pred(:) = mean(horzcat(refAbove(2:N+1), refLeft(2:N+1)));
    
elseif dirMode==1
    %%%%%%%%%%%%%%%%%%%% PLANAR MODE %%%%%%%%%%%%%%%%%%%%
    for x=1:N
        for y=1:N
            PV = (N-y)*refAbove(x+1)+y*refLeft(N+2);
            PH = (N-x)*refLeft(y+1) + x*refAbove(N+2);
            B_pred(y,x) = bitshift(PV+PH+N, -(log2(N)+1), 'int32');
        end
    end

elseif dirMode>=2
    %%%%%%%%%%%%%%%%%%%% ANGULAR PREDICTION %%%%%%%%%%%%%%%%%%%%
    bIsModeVer = (dirMode >= 18);
    if bIsModeVer
        intraPredAngleMode = dirMode - VER_IDX;
    else
        intraPredAngleMode = -(dirMode - HOR_IDX);
    end
    absAngMode = abs(intraPredAngleMode);
    if intraPredAngleMode<0
        signAng = -1;
    else
        signAng = 1;
    end
    
    % Set bitshifts and scale the angle parameter to block size
    angTable = [0, 2, 5, 9, 13, 17, 21, 26, 32];
    invAngTable = [0, 4096, 1638, 910, 630, 482, 390, 315, 256];
    invAngle = invAngTable(absAngMode+1); % +1 for Matlab indexing
    absAng = angTable(absAngMode+1); % +1 for Matlab indexing
    intraPredAngle = signAng*absAng;
    
    % Initialize the Main and Side reference array.
    if intraPredAngle < 0
        if bIsModeVer
            refMain = refAbove(1:N+1);
            refSide = refLeft(1:N+1);
        else
            refMain = refLeft(1:N+1);
            refSide = refAbove(1:N+1);
        end
        
        % Extend the Main reference to the left.
        count = abs(bitshift(N*intraPredAngle, -5, 'int32')+1);
        numb_neg_ind = count;
        tempMain = zeros(1,count);
        invAngleSum = 128;
        for k = -1:-1:bitshift(N*intraPredAngle, -5, 'int32')+1
            invAngleSum = invAngleSum+invAngle;
            tempMain(count) = refSide(bitshift(invAngleSum, -8, 'int32')+1); % +1 per diversa indicizzazione Matlab
            count = count - 1;
        end
        refMain = horzcat(tempMain, refMain);
    else
        dumbNumb = -0.1;
        if bIsModeVer
            refMain = horzcat(refAbove, dumbNumb);
            %         refSide = refLeft;
        else
            refMain = horzcat(refLeft, dumbNumb);
            %         refSide = refAbove;
        end
        numb_neg_ind = 0;
    end
    % numb_neg_ind contiene il numero di eventuali indici negativi in refMain
    
    if bIsModeVer
        for x=1:N
            for y=1:N
                cy = bitshift(y*intraPredAngle, -5, 'int32');
                wy = bitand(y*intraPredAngle, 31, 'int32');
                i = x+cy+1+numb_neg_ind; % +1 per diversa indicizzazione Matlab e count tiene conto di eventuali indici negativi di refMain
                B_pred(y,x) = bitshift((32-wy)*refMain(i)+wy*refMain(i+1)+16, -5, 'int32');
            end
        end
    else
        for x=1:N
            for y=1:N
                cy = bitshift(y*intraPredAngle, -5, 'int32');
                wy = bitand(y*intraPredAngle, 31, 'int32');
                i = x+cy+1+numb_neg_ind; % +1 per diversa indicizzazione Matlab e count tiene conto di eventuali indici negativi di refMain
                B_pred(x,y) = bitshift((32-wy)*refMain(i)+wy*refMain(i+1)+16, -5, 'int32');
            end
        end
    end
    
end

% For debug
% ver = zeros(2*N+1);
% ver(2:N+1, 2:N+1) = B_pred;
% ver(1,:) = refAbove;
% ver(:,1) = refLeft;