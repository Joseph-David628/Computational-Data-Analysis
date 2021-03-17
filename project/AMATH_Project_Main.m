
%load each channel where each channel is a different brain area
%FYI all channels are the same length in time as they were taken from the same session!
%"load channel" data and code inside it like "correct phase" are from the open-ephys git repository
cd = '/Users/joseph/Courses/amath582/amath582/project/';
ch1 = load_channel_data(cd); %mPFC  channel 13
ch2 = load_channel_data(cd); %dHPC  channel 51
ch3 = load_channel_data(cd); %vHPC  channel 44
ds = 30;    
% a simple way to limit the frequency content is to down sample our data. If we sample to 200 samples per second we would be able to analyze 100hz
% the lower limit of our frequency domain can be resitricted based on the window size we use
try
    srate = master_data.evinfo.header.sampleRate;
catch
    % if timestamp/behavior data isn't loaded, assume typical srate
    srate=30000;
end
dsrate = srate/ds;    % downsampling procedure

ds_ch1 = ch1.chdata(1:ds:end); 
ds_ch2 = ch2.chdata(1:ds:end);
ds_ch3 = ch3.chdata(1:ds:end);
ds_tstamps = ch1.chtimestamps(1:ds:end);

%the filters below may be unnecessary if we downsample our data far enough
%however im sure there is a 60Hz frequency from electronics in the room we
%will want to fitler out of each signal otherwise we will have high 60hz
%coherence unrelated to neural activity

ch1_filt = ephys_bpfilt(ds_ch1, dsrate, 5, 200, 'low');
ch2_filt = ephys_bpfilt(ds_ch2, dsrate, 5, 200, 'low');
ch3_filt = ephys_bpfilt(ds_ch3, dsrate, 5, 200, 'low');

%%  visualizing the raw and filtered signal traces for all 3 channels
figure(1)
share_ax1 = subplot(3,2,[1,2]); hold on
plot(share_ax1,ds_tstamps, ds_ch1); hold on
plot(share_ax1, ds_tstamps, ch1_filt, 'LineWidth', 1.5);hold on
set(gca, 'XTick', [])
share_ax2 = subplot(3,2,[3,4]); hold on
plot(share_ax2, ds_tstamps, ds_ch2); hold on
plot(share_ax2, ds_tstamps, ch2_filt, 'LineWidth', 1.5);hold off
share_ax3 = subplot(3,2,[5,6]); hold on
plot(share_ax3, ds_tstamps, ds_ch3); hold on
plot(share_ax3, ds_tstamps, ch3_filt, 'LineWidth', 1.5);hold on

linkaxes([share_ax1 share_ax2 share_ax3], 'x')
hold off

%% testing

fileID = fopen('beh_table.csv');
fgetl(fileID);
epochs = textscan(fileID, '%u8 %u8 %s %u16 %u16 %u16 %f', 'delimiter', ',');
fclose(fileID);

n = 128214;
choice_epoch = zeros(1,n);
delay_epoch = zeros(1,n);
for j = 1 : n 
    if strcmp(epochs{3}{j}, 'choice')
        choice_epoch(j) = 1;
    elseif strcmp(epochs{3}{j}, 'delay')
        delay_epoch(j)=1;
    end
end

start_choice=zeros(1,1);
end_choice=zeros(1,1);
start_delay=zeros(1,1);
end_delay=zeros(1,1);

for j = 2 : n-1
    if choice_epoch(j) - choice_epoch(j-1) == 1
        start_choice(end+1) = epochs{7}(j)-67.74636666666667;
    elseif choice_epoch(j+1) - choice_epoch(j) == -1
        end_choice(end+1) = epochs{7}(j)-67.74636666666667;
    end
    
    if delay_epoch(j) - delay_epoch(j-1) == 1
        start_delay(end+1) = epochs{7}(j)-67.74636666666667;
    elseif delay_epoch(j+1) - delay_epoch(j) == -1
        end_delay(end+1) = epochs{7}(j)-67.74636666666667;
    end
end
start_choice(1)=[];
end_choice(1)=[];
start_delay(1)=[];
end_delay(1)=[];


dsrate = 1000;
m = 4096;
k = dsrate*(0:(m/2))/m;
ch1_power_ave = zeros(m,1);
temp = 0;

for j = 1 : length(start_choice)
    Start = floor(start_choice(j)*dsrate);
    End = floor(end_choice(j)*dsrate);
    if (End-Start) > 4096
        continue
    end
    ch1_power = fftshift(abs(fft(ch1_filt(Start:End),m)).^2);
    ch1_power_ave = ch1_power_ave + ch1_power;
    temp = temp+1;
end

ch1_power_ave = ch1_power_ave * (1/temp);

figure(2)
plot(k(1:500),log10(ch1_power_ave(m/2:m/2+499)));

%%channel2

ch2_power_ave = zeros(m,1);

for j = 1 : length(start_choice)
    Start = floor(start_choice(j)*dsrate);
    End = floor(end_choice(j)*dsrate);
    ch2_power = fftshift(abs(fft(ch2_filt(Start:End),m)).^2);
    ch2_power_ave = ch2_power_ave + ch2_power;
end

ch2_power_ave = ch2_power_ave * (1/length(start_choice));
figure(3)
plot(k(1:500),log10(ch2_power_ave(m/2:m/2+499)));


%%channel 3
ch3_power_ave = zeros(m,1);

for j = 1 : length(start_choice)
    Start = floor(start_choice(j)*dsrate);
    End = floor(end_choice(j)*dsrate);
    ch3_power = fftshift(abs(fft(ch3_filt(Start:End),m)).^2);
    ch3_power_ave = ch3_power_ave + ch3_power;
end

ch3_power_ave = ch3_power_ave * (1/length(start_choice));
figure(4)
plot(k(1:500),log10(ch3_power_ave(m/2:m/2+499)));



        
        
        