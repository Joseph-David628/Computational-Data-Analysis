load cam1_2.mat;
load cam2_2.mat;
load cam3_2.mat;

[height width rgb num_frames] = size(vidFrames1_2);

vidFrames1_2 = im2double(vidFrames1_2);
vidFrames2_2 = im2double(vidFrames2_2(:,:,:,1:num_frames));
vidFrames3_2 = im2double(vidFrames3_2(:,:,:,1:num_frames));

vidFramesB1_2 = (zeros(height, width, num_frames));
vidFramesB2_2 = (zeros(height, width, num_frames));
vidFramesB3_2 = (zeros(height, width, num_frames));

%sum RGB for each frame into one, will be used for finding bright point
for j = 1 : num_frames
    vidFramesB1_2(:,:,j) = sum(vidFrames1_2(:,:,:,j),3);
    vidFramesB2_2(:,:,j) = sum(vidFrames2_2(:,:,:,j),3);
    vidFramesB3_2(:,:,j) = sum(vidFrames3_2(:,:,:,j),3);
end

sz = [height width];
for j = 1 : num_frames
    [m1, i1] = max(vidFramesB1_2(:,:,j),[],'all','linear');
    [m2, i2] = max(vidFramesB2_2(:,:,j),[],'all','linear');
    [m3, i3] = max(vidFramesB3_2(:,:,j),[],'all','linear');
    
    [I1,I2] = ind2sub(sz, [i1 i2 i3]);
    x1(j) = I2(2); x2(j) = I2(1); x3(j) = I2(3);
    y1(j) = I1(2); y2(j) = I1(1); y3(j) = I1(3);
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