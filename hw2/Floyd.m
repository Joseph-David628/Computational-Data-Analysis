clear all; close all; clc

figure(1)
[y, Fs] = audioread('Floyd.m4a');
trgnr = length(y)/Fs; % record time in seconds
n = length(y);
t = (1:n)/Fs;

k = (2*pi/trgnr) * [0:(n/2)-1 -n/2:-1]; k(n)=0; ks = fftshift(k);

a = 1;  %fineness of filter
b = 1; %fineness of slide

a2 = .001; %bass filter width
b2 = 150; %bass filter center
bfilter = exp(-a2*((ksfreq-b2).^2)); %bass filter
yft_spec = [];
ksfreq = ks/(2*pi); %gives frequencies
tslide = 0 : b : t(n);
freq = zeros(1,length(tslide));

for j = 1 : length(tslide)
    filter = exp(-a*((t-tslide(j)).^2));
    yf = y.' .* filter;
    yft = bfilter .* fftshift(fft(yf)); %filter out higher frequencies
    yft_spec = [yft_spec; abs(yft)/max(abs(yft))];
end

figure(3)
pcolor(tslide, ksfreq((n-1)/2 : 1327500), yft_spec(:, (n-1)/2 : 1327500).'), shading interp
set(gca, 'Ylim',[50 150])
title('Comfortably Numb Bass Spectrogram')
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
colormap(hot)

%determine the guitar part
a3 = .00002; b3 = 450;
gfilter = exp(-a3*((ksfreq-b3).^2));
yft_spec = [];
a=100;
b = .3;
tslide = 0 : b : t(n);
for j = 1 : length(tslide)
    filter = exp(-a*((t-tslide(j)).^2));
    yf = y2.' .* filter;
    yt = fftshift(fft(yf)); 
    yft = yt .* gfilter; %filter out lower frequencies   
    yft_spec = [yft_spec; abs(yft)/max(abs(yft))]; 
end

figure(4)
pcolor(tslide, ksfreq((n-1)/2+9000:(n-1)/2+45000), yft_spec(:,(n-1)/2+9000:(n-1)/2+45000).'), shading interp
set(gca, 'Ylim',[150 650])
title('Comfortably Numb Guitar Spectrogram')
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
colormap(hot)

