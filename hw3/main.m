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