% load cam1_1.mat;
% % for j = 1 : 200
% %     A = vidFrames1_1(:,:,:,j);
% %     imshow(A); drawnow
% % end
% load cam2_1.mat;
% load cam3_1.mat;
% 
% [height width rgb num_frames] = size(vidFrames1_1);
% 
% vidFrames1_1 = im2double(vidFrames1_1);
% vidFrames2_1 = im2double(vidFrames2_1(:,:,:,1:num_frames));
% vidFrames3_1 = im2double(vidFrames3_1(:,:,:,1:num_frames));
% 
% vidFramesB1_1 = (zeros(height, width, num_frames));
% vidFramesB2_1 = (zeros(height, width, num_frames));
% vidFramesB3_1 = (zeros(height, width, num_frames));
% 
% %sum RGB for each frame into one, will be used for finding bright point
% for j = 1 : num_frames
%     vidFramesB1_1(:,:,j) = sum(vidFrames1_1(:,:,:,j),3);
%     vidFramesB2_1(:,:,j) = sum(vidFrames2_1(:,:,:,j),3);
%     vidFramesB3_1(:,:,j) = sum(vidFrames3_1(:,:,:,j),3);
% end
% 
% [x1, y1] = findlight(vidFramesB1_1,num_frames,height,width);
% [x2, y2] = findlight(vidFramesB2_1,num_frames,height,width);
% [x3, y3] = findlight(vidFramesB3_1,num_frames,height,width);
% 
% Y = diagonalize(x1,y1,x2,y2,x3,y3,1);

%take difference between frames, largest change will be brightest point
for j = 1 : num_frames-1
    vidFramesC1_1(:,:,j) = vidFramesB1_1(:,:,j+1) - vidFramesB1_1(:,:,j);
    vidFramesC2_1(:,:,j) = vidFramesB2_1(:,:,j+1) - vidFramesB2_1(:,:,j);
    vidFramesC3_1(:,:,j) = vidFramesB3_1(:,:,j+1) - vidFramesB3_1(:,:,j);
end

[x1, y1] = findlight(vidFramesC1_1,num_frames-1,height,width);
[x2, y2] = findlight(vidFramesC2_1,num_frames-1,height,width);
[x3, y3] = findlight(vidFramesC3_1,num_frames-1,height,width);

Y = diagonalize(x1,y1,x2,y2,x3,y3,2);

vidFramesE1 = zeros(height-1, width-1, num_frames);
vidFramesE2 = zeros(height-1, width-1, num_frames);
vidFramesE3 = zeros(height-1, width-1, num_frames);

%take sum of pixels above and below
for k = 1 : num_frames
    for j = 1 : width-1
        for i = 1 : height-1
            vidFramesE1(i,j,k) = vidFramesB1_1(i,j,k)+vidFramesB1_1(i+1,j,k) + vidFramesB1_1(i,j+1,k);
            vidFramesE2(i,j,k) = vidFramesB2_1(i,j,k)+vidFramesB2_1(i+1,j,k) + vidFramesB2_1(i,j+1,k);
            vidFramesE3(i,j,k) = vidFramesB3_1(i,j,k)+vidFramesB3_1(i+1,j,k) + vidFramesB3_1(i,j+1,k);
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


