clear all; close all; clc

%% load video into matrix X and store video attributes
vid = VideoReader('monte_carlo_low.mp4');

numFrames = floor(vid.NumFrames/2);
duration = vid.Duration/2;
height = vid.Height;
width = vid.Width;

t = linspace(0, duration, numFrames); dt = t(2)-t(1);
X=zeros(width*height,numFrames);
for j = 1 : numFrames
    X(:,j) = (reshape(im2double(rgb2gray(readFrame(vid))),width*height,1));
end
clear vid;

%% find singular value spectrum
[u,s,v] = svd(X,'econ');
figure(1)
plot(diag(s)/sum(diag(s)),'Linewidth',[2])
title('Relative Strength of Singular Values for SkiDrop.mp4','FontSize',16)
xlabel('Singular Values','FontSize',16); ylabel('Relative Strength','FontSize',16)
clear u; clear s; clear v;

%% from singular values, determine how many modes r to take and compute DMD
r=3;
X1 = X(:,1:end-1); X2 = X(:,2:end);
[U2,Sigma2,V2] = svd(X1,'econ');
U = U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);
clear U2; clear Sigma2; clear V2;

Atilde = U'*X2*V/Sigma;    
[W,D] = eig(Atilde);    
Phi = X2*V/Sigma*W;
    
mu = diag(D);
omega = log(mu)/dt;

clear U; clear Sigma; clear V;

u0=X(:,1);
y0 = Phi\u0;  % pseudo-inverse initial conditions
u_modes = zeros(r,length(t));
for iter = 1:length(t)
     u_modes(:,iter) =(y0.*exp(omega*t(iter)));
end
u_dmd = Phi*u_modes;

%% look at omega and determine which mode has smallest norm, use as background mode
u_dmd_lr = y0(1) * Phi(:,1) * exp(omega(1)*t);
u_dmd_sparse = u_dmd - abs(u_dmd_lr);

R = zeros(height*width,numFrames);
for i = 1 : height*width
    for j = 1 : numFrames
        if u_dmd_sparse(i,j) < 0
            R(i,j) = u_dmd_sparse(i,j);
        end
    end
    endfr

u_dmd_lr = abs(u_dmd_lr)+ R;
u_dmd_sparse = u_dmd_sparse - R;

clear u_modes; clear y0; clear u0; clear mu; clear D; clear W; clear Atilde;

for j = 120
    %minPix = min(real(u_dmd_sparse(:,j))); maxPix = max(real(u_dmd_sparse(:,j)));
    figure(2)
    imshow(reshape(real(u_dmd_sparse(:,j)),height,width), [])
    figure(3)
    imshow(reshape(real(u_dmd_lr(:,j)),height,width), [])
end




