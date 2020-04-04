close all
clear all
clc

SineSw = load('F16Data_SineSw_Level3.mat');
FullMSine = load('F16Data_FullMSine_Level5.mat');
OddMSine = load('F16Data_SpecialOddMSine_Level2.mat');

fs1 = SineSw.Fs;
fs2 = FullMSine.Fs;
fs3 = OddMSine.Fs;

N1 = 108477;
N2 = 8192;
N3 = 2*8192;

freq1 = 0:fs1/N1:fs1-fs1/N1;
freq2 = 0:fs2/N2:fs2-fs2/N2;
freq3 = 0:fs3/N3:fs3-fs3/N3;

figure; hold on;
plot(freq1,squeeze(db(fft(SineSw.Acceleration(1,:)))))
plot(freq2,squeeze(db(fft(FullMSine.Acceleration(1,end-N2+1:end)))))
plot(freq3,squeeze(db(fft(OddMSine.Acceleration(1,1,end-N3+1:end)))))
