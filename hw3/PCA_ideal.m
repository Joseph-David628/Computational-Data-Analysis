load cam1_1.mat;
load cam2_1.mat;
load cam3_1.mat;

[height width rgb num_frames] = size(vidFrames1_1);

vidFrames1_1 = im2double(vidFrames1_1);
vidFrames2_1 = im2double(vidFrames2_1(:,:,:,1:num_frames));
vidFrames3_1 = im2double(vidFrames3_1(:,:,:,1:num_frames));

vidFramesB1_1 = (zeros(height, width, num_frames));
vidFramesB2_1 = (zeros(height, width, num_frames));
vidFramesB3_1 = (zeros(height, width, num_frames));

%sum RGB for each frame into one, will be used for finding bright point
% for j = 1 : 3
%     vidFramesB1_1 = vidFramesB1_1 + vidFrames1_1(:,:,j,:);
%     vidFramesB2_1 = vidFramesB2_1 + vidFrames2_1(:,:,j,:);
%     vidFramesB3_1 = vidFramesB3_1 + vidFrames3_1(:,:,j,:);
% end

%sum RGB for each frame into one, will be used for finding bright point
for j = 1 : num_frames
    vidFramesB1_1(:,:,j) = sum(vidFrames1_1(:,:,:,j),3);
    vidFramesB2_1(:,:,j) = sum(vidFrames2_1(:,:,:,j),3);
    vidFramesB3_1(:,:,j) = sum(vidFrames3_1(:,:,:,j),3);
end

sz = [height width];
for j = 1 : num_frames
    [m1, i1] = max(vidFramesB1_1(:,:,j),[],'all','linear');
    [m2, i2] = max(vidFramesB2_1(:,:,j),[],'all','linear');
    [m3, i3] = max(vidFramesB3_1(:,:,j),[],'all','linear');
    
    [I1,I2] = ind2sub(sz, [i1 i2 i3]);
    x1(j) = I2(1); x2(j) = I2(2); x3(j) = I2(3);
    y1(j) = I1(1); y2(j) = I1(2); y3(j) = I1(3);
end

X = [x1; y1; x2; y2; x3; y3];
[m,n] = size(X);
mn = mean(X,2);
X = X - repmat(mn,1,n);

Cx = (1/(n-1))*X*X';
[V,D] = eig(Cx);
lambda = diag(D);

[dummy, m_arrange] = sort(-1*lambda);
lambda = lambda(m_arrange);
V=V(:,m_arrange);

Y=V'*X;

figure(1)
for j =1 : 6
    subplot(2,3,j), plot(Y(j,:))
end


