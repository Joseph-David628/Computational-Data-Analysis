% clear all; close all; clc
% 
% load cam1_3.mat;
% load cam2_3.mat;
% load cam3_3.mat;
% 
% [height width rgb num_frames1] = size(vidFrames1_3);
% [height width rgb num_frames2] = size(vidFrames2_3);
% [height width rgb num_frames3] = size(vidFrames3_3);
% 
% num_frames = min([num_frames1 num_frames2 num_frames3]);
% 
% %shorten videos to length of shortest one
% vidFrames1_3 = vidFrames1_3(:,:,:,1:num_frames);
% vidFrames2_3 = vidFrames2_3(:,:,:,1:num_frames);
% vidFrames3_3 = vidFrames3_3(:,:,:,1:num_frames);
% 
% %vidFramesBj = sum over rgb axis
% vidFramesB1 = zeros(height, width, num_frames);
% vidFramesB2 = zeros(height, width, num_frames);
% vidFramesB3 = zeros(height, width, num_frames);
% 
% %sum rgb to help find brightest pixel
% for j = 1 : num_frames
%     vidFramesB1(:,:,j) = sum(vidFrames1_3(:,:,:,j),3);
%     vidFramesB2(:,:,j) = sum(vidFrames2_3(:,:,:,j),3);
%     vidFramesB3(:,:,j) = sum(vidFrames3_3(:,:,:,j),3);
% end
% 
% [x1, y1] = findlight(vidFramesB1,num_frames);
% [x2, y2] = findlight(vidFramesB2,num_frames);
% [x3, y3] = findlight(vidFramesB3,num_frames);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,1);

% vidFramesC1 = zeros(height, width, num_frames);
% vidFramesC2 = zeros(height, width, num_frames);
% vidFramesC3 = zeros(height, width, num_frames);
% 
%vidFramesCj = gradient of vidFramesBj 
% for j = 1 : num_frames
%     [vFx, vFy] = gradient(vidFramesB1(:,:,j));
%     vidFramesC1(:,:,j) = abs(vFx) + abs(vFy);
%     [vFx, vFy] = gradient(vidFramesB2(:,:,j));
%     vidFramesC2(:,:,j) = abs(vFx) + abs(vFy);
%     [vFx, vFy] = gradient(vidFramesB3(:,:,j));
%     vidFramesC3(:,:,j) = abs(vFx) + abs(vFy);
% end
% 
% [x1, y1] = findlight(vidFramesC1,num_frames);
% [x2, y2] = findlight(vidFramesC2,num_frames);
% [x3, y3] = findlight(vidFramesC3,num_frames);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,2);

% vidFramesD1 = zeros(height, width, num_frames);
% vidFramesD2 = zeros(height, width, num_frames);
% vidFramesD3 = zeros(height, width, num_frames);
% 
% %vidFramesDj = second gradient
% for j = 1 : num_frames
%     [vFx, vFy] = gradient(vidFramesC1(:,:,j));
%     vidFramesD1(:,:,j) = abs(vFx) + abs(vFy);
%     [vFx, vFy] = gradient(vidFramesC2(:,:,j));
%     vidFramesD2(:,:,j) = abs(vFx) + abs(vFy);
%     [vFx, vFy] = gradient(vidFramesC3(:,:,j));
%     vidFramesD3(:,:,j) = abs(vFx) + abs(vFy);
% end
% 
% [x1, y1] = findlight(vidFramesD1,num_frames);
% [x2, y2] = findlight(vidFramesD2,num_frames);
% [x3, y3] = findlight(vidFramesD3,num_frames);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,3);

vidFramesE1 = zeros(height-1, width-1, num_frames);
vidFramesE2 = zeros(height-1, width-1, num_frames);
vidFramesE3 = zeros(height-1, width-1, num_frames);

%take difference between pixel above and below, then sum
for k = 1 : num_frames
    for j = 1 : width-1
        for i = 1 : height-1
            vidFramesE1(i,j,k) = vidFramesB1(i,j,k)+vidFramesB1(i+1,j,k) + vidFramesB1(i,j+1,k);
            vidFramesE2(i,j,k) = vidFramesB2(i,j,k)+vidFramesB2(i+1,j,k) + vidFramesB2(i,j+1,k);
            vidFramesE3(i,j,k) = vidFramesB3(i,j,k)+vidFramesB3(i+1,j,k) + vidFramesB3(i,j+1,k);
        end
    end
end

[x1, y1] = findlight(vidFramesE1,num_frames,height-1,width-1);
[x2, y2] = findlight(vidFramesE2,num_frames,height-1,width-1);
[x3, y3] = findlight(vidFramesE3,num_frames,height-1,width-1);

Y = diagonalize(x1,y1,x2,y2,x3,y3,4);

%find position of brightest pixel
function [x,y] = findlight(vidFrames,num_frames,height,width)
    x = zeros(1,num_frames); y = zeros(1,num_frames);
    for j = 1 : num_frames
        [~, i] = max(vidFrames(:,:,j),[],'all','linear');
        [y(j),x(j)] = ind2sub([height width], i);
    end
end

%perform diagonalization and projection
function Y = diagonalize(x1,y1,x2,y2,x3,y3,k)
    X = [x1; y1; x2; y2; x3; y3];
    [~,n] = size(X);
    mn = mean(X,2);
    X = X - repmat(mn,1,n);

    Cx = (1/(n-1))*X*X';
    [V,D] = eig(Cx);
    lambda = diag(D);

    [~, m_arrange] = sort(-1*lambda);
    lambda = lambda(m_arrange);
    V=V(:,m_arrange);

    Y=V'*X;

    figure(k)
    for j = 1 : 6
        subplot(2,3,j), plot(Y(j,:))
    end
end