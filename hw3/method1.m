%load all video data as 4D arrays vidFramesi_j
load cam1_1.mat;
load cam1_2.mat;
load cam1_3.mat;
load cam1_4.mat;
load cam2_1.mat;
load cam2_2.mat;
load cam2_3.mat;
load cam2_4.mat;
load cam3_1.mat;
load cam3_2.mat;
load cam3_3.mat;
load cam3_4.mat;
% 
%this code shortens videos in each experiment to length of shortest
%video
[height width rgb num_frames1_1] = size(vidFrames1_1);
[height width rgb num_frames2_1] = size(vidFrames2_1);
[height width rgb num_frames3_1] = size(vidFrames3_1);
num_frames1 = min([num_frames1_1 num_frames2_1 num_frames3_1]);
vidFrames1_1 = im2double(vidFrames1_1(:,:,:,1:num_frames1));
vidFrames2_1 = im2double(vidFrames2_1(:,:,:,1:num_frames1));
vidFrames3_1 = im2double(vidFrames3_1(:,:,:,1:num_frames1));

[height width rgb num_frames1_2] = size(vidFrames1_2);
[height width rgb num_frames2_2] = size(vidFrames2_2);
[height width rgb num_frames3_2] = size(vidFrames3_2);
num_frames2 = min([num_frames1_2 num_frames2_2 num_frames3_2]);
vidFrames1_2 = im2double(vidFrames1_2(:,:,:,1:num_frames2));
vidFrames2_2 = im2double(vidFrames2_2(:,:,:,1:num_frames2));
vidFrames3_2 = im2double(vidFrames3_2(:,:,:,1:num_frames2));

[height width rgb num_frames1_3] = size(vidFrames1_3);
[height width rgb num_frames2_3] = size(vidFrames2_3);
[height width rgb num_frames3_3] = size(vidFrames3_3);
num_frames3 = min([num_frames1_3 num_frames2_3 num_frames3_3]);
vidFrames1_3 = im2double(vidFrames1_3(:,:,:,1:num_frames3));
vidFrames2_3 = im2double(vidFrames2_3(:,:,:,1:num_frames3));
vidFrames3_3 = im2double(vidFrames3_3(:,:,:,1:num_frames3));

[height width rgb num_frames1_4] = size(vidFrames1_4);
[height width rgb num_frames2_4] = size(vidFrames2_4);
[height width rgb num_frames3_4] = size(vidFrames3_4);
num_frames4 = min([num_frames1_4 num_frames2_4 num_frames3_4]);
vidFrames1_4 = im2double(vidFrames1_4(:,:,:,1:num_frames4));
vidFrames2_4 = im2double(vidFrames2_4(:,:,:,1:num_frames4));
vidFrames3_4 = im2double(vidFrames3_4(:,:,:,1:num_frames4));

%sum RGB axes into one
vidFrames1_1 = sumRGB(vidFrames1_1,height,width,num_frames1);
vidFrames2_1 = sumRGB(vidFrames2_1,height,width,num_frames1);
vidFrames3_1 = sumRGB(vidFrames3_1,height,width,num_frames1);

vidFrames1_2 = sumRGB(vidFrames1_2,height,width,num_frames2);
vidFrames2_2 = sumRGB(vidFrames2_2,height,width,num_frames2);
vidFrames3_2 = sumRGB(vidFrames3_2,height,width,num_frames2);

vidFrames1_3 = sumRGB(vidFrames1_3,height,width,num_frames3);
vidFrames2_3 = sumRGB(vidFrames2_3,height,width,num_frames3);
vidFrames3_3 = sumRGB(vidFrames3_3,height,width,num_frames3);

vidFrames1_4 = sumRGB(vidFrames1_4,height,width,num_frames4);
vidFrames2_4 = sumRGB(vidFrames2_4,height,width,num_frames4);
vidFrames3_4 = sumRGB(vidFrames3_4,height,width,num_frames4);

%first method: simply take largest value as position of can
[x1,y1] = findlight(vidFrames1_1,height,width,num_frames1);
[x2,y2] = findlight(vidFrames2_1,height,width,num_frames1);
[x3,y3] = findlight(vidFrames3_1,height,width,num_frames1);
PCA(x1,y1,x2,y2,x3,y3,1,1);

[x1,y1] = findlight(vidFrames1_2,height,width,num_frames2);
[x2,y2] = findlight(vidFrames2_2,height,width,num_frames2);
[x3,y3] = findlight(vidFrames3_2,height,width,num_frames2);
PCA(x1,y1,x2,y2,x3,y3,1,2);

[x1,y1] = findlight(vidFrames1_3,height,width,num_frames3);
[x2,y2] = findlight(vidFrames2_3,height,width,num_frames3);
[x3,y3] = findlight(vidFrames3_3,height,width,num_frames3);
PCA(x1,y1,x2,y2,x3,y3,1,3);

[x1,y1] = findlight(vidFrames1_4,height,width,num_frames4);
[x2,y2] = findlight(vidFrames2_4,height,width,num_frames4);
[x3,y3] = findlight(vidFrames3_4,height,width,num_frames4);
PCA(x1,y1,x2,y2,x3,y3,1,4);

%method 2: take abs of gradient in both directions and sum
[x1,y1] = findlight(grad_space(vidFrames1_1,height,width,num_frames1),height,width,num_frames1);
[x2,y2] = findlight(grad_space(vidFrames2_1,height,width,num_frames1),height,width,num_frames1);
[x3,y3] = findlight(grad_space(vidFrames3_1,height,width,num_frames1),height,width,num_frames1);
PCA(x1,y1,x2,y2,x3,y3,2,1);

[x1,y1] = findlight(grad_space(vidFrames1_2,height,width,num_frames2),height,width,num_frames2);
[x2,y2] = findlight(grad_space(vidFrames2_2,height,width,num_frames2),height,width,num_frames2);
[x3,y3] = findlight(grad_space(vidFrames3_2,height,width,num_frames2),height,width,num_frames2);
PCA(x1,y1,x2,y2,x3,y3,2,2);

[x1,y1] = findlight(grad_space(vidFrames1_3,height,width,num_frames3),height,width,num_frames3);
[x2,y2] = findlight(grad_space(vidFrames2_3,height,width,num_frames3),height,width,num_frames3);
[x3,y3] = findlight(grad_space(vidFrames3_3,height,width,num_frames3),height,width,num_frames3);
PCA(x1,y1,x2,y2,x3,y3,2,3);

[x1,y1] = findlight(grad_space(vidFrames1_4,height,width,num_frames4),height,width,num_frames4);
[x2,y2] = findlight(grad_space(vidFrames2_4,height,width,num_frames4),height,width,num_frames4);
[x3,y3] = findlight(grad_space(vidFrames3_4,height,width,num_frames4),height,width,num_frames4);
PCA(x1,y1,x2,y2,x3,y3,2,4);

method 3: change in pixel brightness in time
[x1,y1] = findlight(grad_time(vidFrames1_1,height,width,num_frames1),height, width, num_frames1-1);
[x2,y2] = findlight(grad_time(vidFrames2_1,height,width,num_frames1),height, width, num_frames1-1);
[x3,y3] = findlight(grad_time(vidFrames3_1,height,width,num_frames1),height, width, num_frames1-1);
PCA(x1,y1,x2,y2,x3,y3,3,1);

[x1,y1] = findlight(grad_time(vidFrames1_2,height,width,num_frames2),height, width, num_frames2-1);
[x2,y2] = findlight(grad_time(vidFrames2_2,height,width,num_frames2),height, width, num_frames2-1);
[x3,y3] = findlight(grad_time(vidFrames3_2,height,width,num_frames2),height, width, num_frames2-1);
PCA(x1,y1,x2,y2,x3,y3,3,2);

[x1,y1] = findlight(grad_time(vidFrames1_3,height,width,num_frames3),height, width, num_frames3-1);
[x2,y2] = findlight(grad_time(vidFrames2_3,height,width,num_frames3),height, width, num_frames3-1);
[x3,y3] = findlight(grad_time(vidFrames3_3,height,width,num_frames3),height, width, num_frames3-1);
PCA(x1,y1,x2,y2,x3,y3,3,3);

[x1,y1] = findlight(grad_time(vidFrames1_4,height,width,num_frames4),height, width, num_frames4-1);
[x2,y2] = findlight(grad_time(vidFrames2_4,height,width,num_frames4),height, width, num_frames4-1);
[x3,y3] = findlight(grad_time(vidFrames3_4,height,width,num_frames4),height, width, num_frames4-1);
PCA(x1,y1,x2,y2,x3,y3,3,4);

%method 4: sum neighbors
[x1,y1] = findlight(sum_neighbors(vidFrames1_1,height,width,num_frames1), height-1, width-1, num_frames1);
[x2,y2] = findlight(sum_neighbors(vidFrames2_1,height,width,num_frames1), height-1, width-1, num_frames1);
[x3,y3] = findlight(sum_neighbors(vidFrames3_1,height,width,num_frames1), height-1, width-1, num_frames1);
PCA(x1,y1,x2,y3,x3,y3,4,1);

[x1,y1] = findlight(sum_neighbors(vidFrames1_2,height,width,num_frames2), height-1, width-1, num_frames2);
[x2,y2] = findlight(sum_neighbors(vidFrames2_2,height,width,num_frames2), height-1, width-1, num_frames2);
[x3,y3] = findlight(sum_neighbors(vidFrames3_2,height,width,num_frames2), height-1, width-1, num_frames2);
PCA(x1,y1,x2,y3,x3,y3,4,2);

[x1,y1] = findlight(sum_neighbors(vidFrames1_3,height,width,num_frames3), height-1, width-1, num_frames3);
[x2,y2] = findlight(sum_neighbors(vidFrames2_3,height,width,num_frames3), height-1, width-1, num_frames3);
[x3,y3] = findlight(sum_neighbors(vidFrames3_3,height,width,num_frames3), height-1, width-1, num_frames3);
PCA(x1,y1,x2,y3,x3,y3,4,3);

[x1,y1] = findlight(sum_neighbors(vidFrames1_4,height,width,num_frames4), height-1, width-1, num_frames4);
[x2,y2] = findlight(sum_neighbors(vidFrames2_4,height,width,num_frames4), height-1, width-1, num_frames4);
[x3,y3] = findlight(sum_neighbors(vidFrames3_4,height,width,num_frames4), height-1, width-1, num_frames4);
PCA(x1,y1,x2,y3,x3,y3,4,4);


% --------------- function definitions -------------------------------

%method 1: this function sums the RGB axis into one
function x = sumRGB(vidFrames,height,width,num_frames)
    x = zeros(height,width,num_frames);
    for j =1 : num_frames
        x(:,:,j) = sum(vidFrames(:,:,:,j),3);
    end
end 

%method 2: this function returns the gradient
function x = grad_space(vidFrames, height, width, num_frames)
    x = zeros(height, width, num_frames);
    for j = 1 : num_frames
        [vFx, vFy] = gradient(vidFrames(:,:,j));
        x(:,:,j) = abs(vFx)+abs(vFy);
    end
end

%method 3: change in time
function x = grad_time(vidFrames, height, width, num_frames)
    x = zeros(height, width, num_frames-1);
    for j = 2 : num_frames
        x(:,:,j-1) = vidFrames(:,:,j) - vidFrames(:,:,j-1);
    end
end

%method 4: sum neighbors
function x = sum_neighbors(vidFrames, height, width, num_frames)
    x = zeros(height-1,width-1,num_frames);
    for j = 1 : num_frames
        for i = 1 : height-1
            for k = 1 : width-1
                x(i,k,j) = vidFrames(i,k,j) + vidFrames(i+1,k,j) + vidFrames(i,k+1,j);
            end
        end
    end 
end

%find position of brightest pixel
function [x,y] = findlight(vidFrames,height,width,num_frames)
    x = zeros(1,num_frames); y = zeros(1,num_frames);
    for j = 1 : num_frames
        [~, i] = max(vidFrames(:,:,j),[],'all','linear');
        [y(j),x(j)] = ind2sub([height width], i);
    end
end

%perform diagonalization and projection
function Y = PCA(x1,y1,x2,y2,x3,y3,k,r)
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
    subplot(2,2,r), plot(Y(2,:))
    
   if r == 1
       xlabel('I. Ideal')
   elseif r == 2
       xlabel('II. Noisy')
   elseif r == 3
       xlabel('III. Horizontal')
   else
       xlabel('IV. Horizontal & Rotation')
   end
    
end