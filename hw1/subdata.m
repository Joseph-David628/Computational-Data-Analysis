clear all; close all; clc

load subdata.mat; % imports the data as the 262144 x 49 (space by time) matrix

L=10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y=x; z=x;
k = (2*pi/(2*L)) * [0 : (n/2 - 1) -n/2 : -1]; ks = fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%first, average over the frequencies to find the center frequency
Unfave = zeros(n,n,n);

for j = 1 : 49
    Un = reshape(subdata(:,j),n,n,n);
    Utn = fftshift(fftn(Un));
    Unfave = Unfave + Utn;
end

[M,I] = max(abs(Unfave), [], 'all', 'linear'); 

figure(1)
isosurface(Kx,Ky,Kz, abs(Unfave)/M, 0.6), grid on
title('Average Fourier Transform (Iso value=0.6)'); 
xlabel('x wavenumbers'); ylabel('y wavenumbers'); zlabel('z wavenumbers');
ax = gca; ax.FontSize = 16;
xlim([-10 10]); ylim([-10 10]); zlim([-10 10]);

sz = [n n n];
[I1, I2, I3] = ind2sub(sz,I); %contains the indices of the max element in Unfave

%create filter centered around the center frequency
%center frequency is at (ks(I1), ks(I2), ks(I3))
c = .9;
filter = sech(c*(Kx - ks(I2))) .* sech(c*(Ky - ks(I1))) .* sech(c*(Kz - ks(I3)));

%go through each realization of subdata and Fourier transfrom
%then apply filter and take inverse Fourier
a = zeros(1,49);
b = zeros(1,49);
c = zeros(1,49);
for j = 1 : 49
    Un = reshape(subdata(:,j),n,n,n);
    Utn = fftshift(fftn(Un));
    Utnf = Utn .* filter;
    U = ifftn(Utnf);
    [M2, J2] = max(abs(U),[],'all', 'linear');
    [b(j), a(j), c(j)] = ind2sub(sz, J2);
end
 
figure(2),
subplot(2,1,1), plot3(x(a), y(b), z(b), '-*'), grid on;
title('Submarine Path in 3D'); xlabel('x'); ylabel('y'); zlabel('z');
ax = gca; ax.FontSize = 16;
xlim([-10 10]); ylim([-10 10]); zlim([-10 10]);
subplot(2,1,2), plot(x(a), y(b), '-*'), grid on;
title('Submarine Path in 2D (aka P-8 Path)'); xlabel('x'); ylabel('y');
ax = gca; ax.FontSize = 16;
xlim([-10 10]); ylim([-10 10]);
