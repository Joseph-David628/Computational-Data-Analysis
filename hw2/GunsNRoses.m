clear all; close all; clc

figure(1)
[y, Fs] = audioread('GNR.m4a');
trgnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine');
% p8 = audioplayer(y,Fs); playblocking(p8);

n = length(y);
t = (1:n)/Fs;

k = (2*pi/trgnr) * [0:(n/2)-1 -n/2:-1]; ks = fftshift(k);
% fy = fftshift(fft(y));
% figure(2)
% plot(ks,fy);

a = 30;  %fineness of filter
b = .125; %fineness of slide
yft_spec = [];
ksfreq = ks*(1/(2*pi));
tslide = b/2 : b : t(n);

freq = zeros(1,length(tslide));

for j = 1 : length(tslide)
    filter = exp(-a*((t-tslide(j)).^2));
    yf = y.' .* filter;
    yft = fft(yf);
    
    yft_spec = [yft_spec; abs(fftshift(yft))/max(abs(yft))];
    
    figure(2)
    subplot(3,1,1), plot(t, y, 'k', t, filter, 'r')
    subplot(3,1,2), plot(t, abs(yf), 'k')
    subplot(3,1,3), plot(ks, abs(fftshift(yft))/max(abs(yft)))
    axis([-5000 5000 0 1])
    
end

figure(3)
pcolor(tslide, ksfreq(n/2+3000:n), yft_spec(:,n/2+3000:n).'), shading interp
set(gca, 'Ylim',[218 5000/(2*pi)])
title('Sweet Child O Mine Spectrogram')
xlabel('Time [sec]'); ylabel('Frequency [Hz]')
colormap(hot)
