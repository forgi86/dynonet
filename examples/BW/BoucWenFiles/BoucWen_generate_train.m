%
% Example of use of the BoucWen_TimeIntegration.p routine.
% 
% J.P. Noel(1), M. Schoukens(2), September 2015.
%
% (1) University of Liege, Belgium.
% (2) Vrije Universiteit Brussel, Belgium.
%

%% Clean workspace 
clear;
clc;

%% Time integration parameters.

fs = 750;               % working sampling frequency.
upsamp = 20;            % upsampling factor to ensure integration accuracy.
fsint = fs*upsamp;      % integration sampling frequency.
h = 1/fsint;            % integration time step.

%% Dataset path
filename = 'train.h5';
data_folder = 'Test signals';

%% Excitation signal design. 

P = 5;                  % number of excitation periods.
N = 8192;               % number of points per period.
Nint = N*upsamp;        % number of points per period during integration.

fmin = 5;               % excitation bandwidth.
fmax = 150;

A = 50;                 % excitation amplitude.

Pfilter = 1;            % extra period to avoid edge effects during low-pass filtering (see line 59).
P = P + Pfilter;

U = zeros(Nint,1);      % definition of the multisine excitation.
fres = fsint/Nint;
exclines = 1:ceil(fmax/fres);
exclines(exclines < floor(fmin/fres)) = [];

U(exclines+1) = exp(complex(0,2*pi*rand(size(exclines))));
u = 2*real(ifft(U));
u = A*u/std(u);
u = repmat(u,[P 1]);

%% Time integration.

y0 = 0;                 % initial conditions.
dy0 = 0;
z0 = 0;

start = tic;
y = BoucWen_NewmarkIntegration(h,u,y0,dy0,z0);
stop = toc(start);disp(['Duration of the time integration: ',num2str(stop),' s.']);

%% Low-pass filtering and downsampling.

drate = factor(upsamp);        % prime factor decomposition.

for k=1:length(drate)
    y = decimate(y,drate(k),'fir');    
end %k

u = downsample(u,upsamp);

%% Removal of the last simulated period to eliminate the edge effects due to the low-pass filter.

y = y(1:(P-1)*N);
u = u(1:(P-1)*N,:);
P = P-1;


%% Prepare data %%
N = length(y);

u = transpose(u);
%% Save results in an hdf file %%


filepath = fullfile(data_folder, filename);
h5create(filepath, '/multisine/u', size(u))
h5write(filepath, '/multisine/u', u)

h5create(filepath, '/multisine/y', size(y))
h5write(filepath, '/multisine/y', y)

h5create(filepath, '/multisine/fs', size(fs))
h5write(filepath, '/multisine/fs', fs)

h5disp(filepath)
