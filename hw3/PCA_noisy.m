load cam1_2.mat;
load cam2_2.mat;
load cam3_2.mat;

[height width rgb num_frames] = size(vidFrames1_2);

%convert matrices to doubles to perform operations
vidFrames1_2 = im2double(vidFrames1_2);
vidFrames2_2 = im2double(vidFrames2_2(:,:,:,1:num_frames));
vidFrames3_2 = im2double(vidFrames3_2(:,:,:,1:num_frames));

vidFramesB1_2 = (zeros(height, width, num_frames));
vidFramesB2_2 = (zeros(height, width, num_frames));
vidFramesB3_2 = (zeros(height, width, num_frames));

%sum RGB for each frame into one, max will be brightest point
for j = 1 : num_frames
    vidFramesB1_2(:,:,j) = sum(vidFrames1_2(:,:,:,j),3);
    vidFramesB2_2(:,:,j) = sum(vidFrames2_2(:,:,:,j),3);
    vidFramesB3_2(:,:,j) = sum(vidFrames3_2(:,:,:,j),3);
end

% [x1, y1] = findlight(vidFramesB1_2,num_frames);
% [x2, y2] = findlight(vidFramesB2_2,num_frames);
% [x3, y3] = findlight(vidFramesB3_2,num_frames);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,1);

% vidFramesC1_2 = (zeros(height, width, num_frames-1));
% vidFramesC2_2 = (zeros(height, width, num_frames-1));
% vidFramesC3_2 = (zeros(height, width, num_frames-1));
% 
% %take difference between frames, largest change will be brightest point
% for j = 1 : num_frames-1
%     vidFramesC1_2(:,:,j) = vidFramesB1_2(:,:,j+1) - vidFramesB1_2(:,:,j);
%     vidFramesC2_2(:,:,j) = vidFramesB2_2(:,:,j+1) - vidFramesB2_2(:,:,j);
%     vidFramesC3_2(:,:,j) = vidFramesB3_2(:,:,j+1) - vidFramesB3_2(:,:,j);
% end
% 
% [x1, y1] = findlight(vidFramesC1_2,num_frames-1);
% [x2, y2] = findlight(vidFramesC2_2,num_frames-1);
% [x3, y3] = findlight(vidFramesC3_2,num_frames-1);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,2);

% vidFramesD1_2 = (zeros(height, width, num_frames-1));
% vidFramesD2_2 = (zeros(height, width, num_frames-1));
% vidFramesD3_2 = (zeros(height, width, num_frames-1));
% 
% %calculate gradient, presumably pixel with biggest change will have the
% %light
% for j = 1 : num_frames-1
%     [vFx1_2,vFy1_2] = gradient(vidFramesB1_2(:,:,j));
%     [vFx2_2,vFy2_2] = gradient(vidFramesB2_2(:,:,j));
%     [vFx3_2,vFy3_2] = gradient(vidFramesB3_2(:,:,j));
%     
%     vidFramesD1_2(:,:,j) = vFx1_2 + vFy1_2;
%     vidFramesD2_2(:,:,j) = vFx2_2 + vFy2_2;
%     vidFramesD3_2(:,:,j) = vFx3_2 + vFy3_2;
% end
% 
% [x1, y1] = findlight(vidFramesD1_2,num_frames-1);
% [x2, y2] = findlight(vidFramesD2_2,num_frames-1);
% [x3, y3] = findlight(vidFramesD3_2,num_frames-1);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,3);

%take difference between pixel above and below, then sum
for k = 1 : num_frames
    for j = 1 : width-1
        for i = 1 : height-1
            vidFramesE1(i,j,k) = vidFramesB1_2(i,j,k)+vidFramesB1_2(i+1,j,k) + vidFramesB1_2(i,j+1,k);
            vidFramesE2(i,j,k) = vidFramesB2_2(i,j,k)+vidFramesB2_2(i+1,j,k) + vidFramesB2_2(i,j+1,k);
            vidFramesE3(i,j,k) = vidFramesB3_2(i,j,k)+vidFramesB3_2(i+1,j,k) + vidFramesB3_2(i,j+1,k);
        end
    end
end

[x1, y1] = findlight(vidFramesE1,num_frames);
[x2, y2] = findlight(vidFramesE2,num_frames);
[x3, y3] = findlight(vidFramesE3,num_frames);

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