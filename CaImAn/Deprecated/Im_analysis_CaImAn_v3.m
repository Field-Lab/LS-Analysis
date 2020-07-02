 
%% Set up utils and data path 

% Choose architecture: 
%   1. Linux
%   2. MacOS 
%   3. Windows
carch = computer('arch'); 


% Set path directive 
if regexpi(carch,'glnx')
    initpath = '/Volumes/dusom_fieldlab/All_Staff/lab/';
    caimanpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/CaImAn/CaImAn-MATLAB-master/'];
    addpath(genpath(caimanpath)); 
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/Internal_utils/']));
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/External_utils/']));
elseif regexpi(carch,'maci') 
    initpath = '/Volumes/dusom_fieldlab/All_Staff/lab/';
    caimanpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/CaImAn/CaImAn-MATLAB-master/'];
    addpath(genpath(caimanpath)); 
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/Internal_utils/']));
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/External_utils/']));
elseif regexpi(carch,'win')
    initpath = 'Z:/lab/';
    caimanpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/CaImAn/CaImAn-MATLAB-master/'];
    addpath(genpath(caimanpath)); 
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/Internal_utils/']));
    addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/External_utils/']));
end


% Set full path for raw data, scripts and analyzed data and figures 
datanam = 'Data_17'; 
datadat = '2020-05-08-0';
whicheye = 'LE'; 
tiffpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/',char(datanam),'/',char(datanam),'.tif'];
fijiprojim = [];
if ~exist([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs'],'dir')
    mkdir([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs']);
end
matfigsavepath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs/'];

% Names of videos 
vid_fullim_roi_fluorescentTrace = [matfigsavepath,'vid_',char(datanam),'_fullim_roi_fluorescentTrace.mp4'];

% Do you have stimulus? 
stim_exist = true;



%% Set up parallel workers, gpu arrays, located mex files and screen preference 

% Set screen preference 
posarr = get(0,'MonitorPositions'); 
if size(posarr,1)>1
    desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
    desktopMainFrame = desktop.getMainFrame;
    lowx = desktopMainFrame.getLocation.x; 
    lowy = desktopMainFrame.getLocation.y;
    if (lowx>posarr(1,3) && lowy<posarr(1,4)) || (lowx<posarr(1,3) && lowy>posarr(1,4))  % on second monitor registered as horizontally shifted
        shift = posarr(2,1:2);
        figh = figure('Visible','off');
        set(figh, 'Units', 'pixels');
        pos = get(figh, 'Position');
        pos = [pos(1:2)+shift, pos(3:4)]; 
        set(0, 'DefaultFigurePosition', pos); 
        close(figh); 
    end
end


% Setup parallel workers 
v = ver; 
popt = any(strcmp({v.Name}, 'Parallel Computing Toolbox')); 
if popt 
    p = gcp('nocreate'); 
    ncores = maxNumCompThreads('automatic'); 
    if ncores > 1 && isempty(p)
        parpool('local', ncores); 
    end
end

% Set up gpu flag (if it exists)
if ~gpuDeviceCount
    gpuflag = true;  
else
    gpuflag = false;  
end
    
% Set up mex files (need to be recompiled if unlinked) 
outstr = which('graph_conn_comp_mex'); 
if isempty(outstr)
    mex -setup C;
    mex -setup C++; 
    mexfilepath = fullfile(caimanpath,'utilities/graph_conn_comp_mex.cpp'); 
    mex('-v','-compatibleArrayDims',mexfilepath, '-outdir',fullfile(caimanpath,'utilities/'));
end

%% Read image file, remove outliers, denoise images 

% Steps: 
%   (1) Read image files and properties 
%   (2) Remove redundant regions of the image (Suggested: select this region after
%       watching the activity in FIJI)
%   (3) Denoise images using either pre-trained deep learning network
%   (4) Identify sequence of images where light was flashed (not implemented yet)

% ****** Implement sparse/Tall arrays for memory efficiency 

% ------------------------------ Step 1 -----------------------------------
% Read image properties 
info = imfinfo(tiffpath);
%num_images = numel(info);
num_images = 600;
bitdepth = unique([info.BitDepth]); 
dims = [info(1).Height info(2).Width]; 
AOI_red{1} = 1:dims(1); % Area of interest (reduced)
AOI_red{2} = 1:dims(2); 

strextrct = info(1).ImageDescription(regexpi(info(1).ImageDescription,'Exposure'):regexpi(info(1).ImageDescription,'Exposure')+18); 
fps = 1/str2double(strextrct(regexpi(strextrct,'=')+2:regexpi(strextrct,'=')+8)); % frame rate (/sec)
decaytconst = 270/1000; % sec (GCaMP7f: from Dana etal. 2018)

% ------------------------------ Step 2 -----------------------------------
% Calcium image files can be big, we want to minimize redundancy in the
% image set for analysis as much as possible 

% Max std projection for identification of Area of Interest 
im_uint16 = zeros([dims num_images],'uint16'); 
parfor_progress(num_images); 
parfor k=1:num_images
    im_uint16(:,:,k) = imread(tiffpath,k); 
    parfor_progress; 
end
parfor_progress(0); 
medZim = median(im_uint16,3);  % Median projection
stdZim = std(double(im_uint16),0,3); % std projection


% Ask user for selecting Area of Interest
hf = figure(1); imshow(medZim,[]); imcontrast; hold on; title('Select the vertices of rectangular Area of Interest'); 
[x,y] = deal([]); 
for i=1:2
    [x(i),y(i)] = ginput(1);
    h = plot(x(i),y(i),'or','markersize',12,'linewidth',3); hold on; 
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
end
x = sort(ceil(x)); y = sort(ceil(y)); 
AOI_x = [x(1) x(2) x(2) x(1) x(1)]; % Area of Interest
AOI_y = [y(1) y(1) y(2) y(2) y(1)]; 
plot(AOI_x,AOI_y,'-y','linewidth',3); legend('Area for analysis'); 
%saveas(hf, [savefigpath,'AOI_selection'], 'fig'); 

% Generate a smaller set of image 
im_uint16_red = im_uint16(min(AOI_y):max(AOI_y),min(AOI_x):max(AOI_x),:); 
dims_red = [size(im_uint16_red,1) size(im_uint16_red,2)];

% ------------------------------ Step 3 -----------------------------------
% Extract radius of putative ROIs - cell bodies, dendritic spines etc. 
% This will be used for background brightness adjustment and by CaImAn (as
% the standard deviation of Gaussian kernel for spatial segmentation; std = 
% half the size of neuron)
% Limitation: Current version allows only 1 size of ROIs. 

hf = figure; 
imshow(medZim,[]); hold on; 
title('Zoom into a region for selecting ROIs, then press Enter!');
zoom(hf,'on'); 
pause(); 
title('Select the outline of an ROI using left click, and press Enter when done!');
zoom reset;
clear x y; 
[x_,y_] = deal([]); 
but = 1; 
while ~isempty(but)
    [x,y,but] = ginput(1);
    x_ = [x_ x];
    y_ = [y_ y];
    plot(x_,y_,'.-y','markersize',8,'linewidth',1); hold on;
end
title('Template ROI is now selected'); 
pshape = polyshape(x_,y_,'simplify',true);
rad = round(sqrt(pshape.area/pi)); % fit a circle to a polygon


% ------------------------------ Step 4 -----------------------------------
% Adjust uneven brightness from stimulus if captured on images 
if stim_exist
    stimremv = questdlg( 'Do you want auto-removal of stimulus brightness from images?','Auto-remove stimulus.',...
    ['Yes'],['No'],['No']);
    if strcmpi(stimremv,'Yes')
        se = strel('disk',rad);
        parfor_progress(size(im_uint16_red,3));
        im_uint16_red_lumadj = uint16(zeros(size(im_uint16_red)));
        parfor ki= 1:size(im_uint16_red,3)
            im_uint16_red_lumadj(:,:,ki) = imtophat(im_uint16_red(:,:,ki), se);
            parfor_progress;
        end
        parfor_progress(0);
        im_uint16_red = im_uint16_red_lumadj;
        medZim = median(im_uint16_red,3); 
        clear im_uint16_red_lumadj; 
    end 
end

% ------------------------------ Step 5 -----------------------------------
% Remove Light Sheet induced stripes from images 
% Method 1 : FFT and inverseFFT of mask
% Method 2 : Multidirectional Stripe Remover based on Nonsubsampled
% Contour Wavelet Transform 


% ------------------------------ Step 6 -----------------------------------
% Denoise images using either (1) Non-local means filtering, or, (2) CNN classifier 

% Denoising image based on Non-local means filtering 
im_doub_touse = zeros([dims_red num_images],'double');
parfor_progress(num_images);
parfor k=1:num_images
    imtemp = cast(im_uint16_red(:,:,k),'double');
    im_doub_touse(:,:,k) = imnlmfilt(imtemp,'SearchWindowSize',15,'ComparisonWindowSize',5);
    parfor_progress;
end
parfor_progress(0);

% 
% if exist(matfigsavepath,'dir')
%     savefig(hf,[matfigsavepath,'define_std_GaussKernel.fig']); 
% else
%     warning('Directory does not exist!! Skipping saving file');
% end
%saveas(hf, [savefigpath,'define_std_GaussKernel'], 'fig'); 



%% Run CaImAn

Y = cast(im_doub_touse,'single'); 
clear im_doub_touse; 
 
%Y = Y - min(Y(:)); 
if ~isa(Y,'single');    Y = single(Y);  end         % convert to single
 
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels
%% Query stimulus 

 
if stim_exist
    stim_def = 'pulse';
    pulse_wid_ms = 100; % ms 
    pulse_wid = pulse_wid_ms/1000; % sec
    fr_per_pulse = round(pulse_wid/(1/fps)); 
    period = 4; % sec 
    minsep = max(period-5*pulse_wid, period/2); 
end
frame_lum = squeeze(sum(sum(Y)))/(size(Y,1)*size(Y,2)); 
[~,rise_time] = findpeaks(frame_lum,fps, 'MinPeakDistance',minsep); 
rise_frame = rise_time.*fps; 

backg = zeros(size(Y,1),size(Y,2),1,'single'); 
parfor_progress(length(rise_frame)-1); 
for k=1:length(rise_frame)-1
    backg = backg + mean(Y(:,:,rise_frame(k)+10*fr_per_pulse:rise_frame(k+1)-10*fr_per_pulse),3);
    parfor_progress; 
end
parfor_progress(0);
backg = backg./(length(rise_frame)-1); 
upper_thresh = 1*mad(backg(:))+median(backg(:));
outlier_loc = find(frame_lum>upper_thresh); 
new_otlvals = zeros(1,length(outlier_loc)); 
for otl=1:length(outlier_loc)
    Y(:,:,outlier_loc(otl)) = backg; 
    new_otlvals(otl) = sum(backg(:))/(size(backg,1)*size(backg,2));
end

figure; subplot(2,1,1); plot(1:num_images,frame_lum,'.k',outlier_loc,frame_lum(outlier_loc),'or'); yll = get(gca,'ylim');
subplot(2,1,2); plot(1:num_images,squeeze(sum(sum(Y)))/(size(Y,1)*size(Y,2)),'.k'); set(gca,'ylim',yll); 


%% Set parameters 

K = 15;                                             % number of components to be found
tau = rad;                                          % std of gaussian kernel (half size of neuron) : Need to be Integer 
p = 2;                                              % order of AR model dynamics
pixels = 1:numel(Y)/size(Y,ndims(Y));               % pixels to include when computing the AR coefs
 
options = CNMFSetParms(...
    'd1',d1,'d2',d2,...                             % dimensionality of the FOV        
    'p',p,...                                       % order of AR dynamics    
    'gSig',tau,...                                  % half size of neurons
    'pixels',pixels,'extract_max',false,...         % pixels used for AR coefficients
    'cluster_pixels',true,'flag_g',true,...         % cluster pixels based on closeness of noise PSD 
    'split_data',true,...                           % reduce memory overhead 
    'merge_thr',0.80,...                            % merge of activity correlation is > 
    'nb',2,...                                      % number of background components    
    'min_SNR',3,...                                 % minimum SNR threshold
    'space_thresh',0.5,...                          % space correlation threshold
    'cnn_thr',0.2,...                               % threshold for CNN classifier 
    'fr',fps,....                                   % frame rate of image acquisition 
    'decay_time',decaytconst,...                    % decay constant of GCaMP signal 
    'sx',12,...                                     % half size of representative patches (default: 16)  - for plotting 
    'ind',[1:10],...                                % choose some example components for saving (or creating movies) 
    'skip_frame', 1,...                             % skip frames when showing the video (default: 1 (no skipping))
    'make_avi', 1,...                               % flag for saving avi video (default: 0)
    'show_background',1,...                         % flag for displaying the background in the denoised panel (default: 1)
    'show_contours',0,...                           % flag for showing the contour plots of the patches in the FoV (default: 0)
    'name',vid_fullim_roi_fluorescentTrace);
    

%% Data pre-processing
 
[P,Y] = preprocess_data(Y,p,options);

%% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

% display centers of found components
Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)

% clear variable name conflict with function name
if strcmp(which('colormap'),'variable'); clear colormap; end 

figure; set(gcf,'position',[235   177   971   599]); 
imshow(Cn,[]); 
axis equal; axis tight; hold all;
scatter(center(:,2),center(:,1),'mo');
title('Center of ROIs found from initialization algorithm');
drawnow;


%% manually refine components (optional)
refine_components = true;  % flag for manual refinement
if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end
    
%% update spatial components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

%% update temporal components
P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

%% classify components (implementation in Pnevmatikakis et.al) 

rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        % this test will keep processes
                                        
%% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1); 
    % components that pass the CNN classifier
end     
                            
%% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% select components

keep = (ind_corr | ind_cnn) & ind_exc;

%% display kept and discarded components
A_keep = A(:,keep);
C_keep = C(keep,:);
figure;
    subplot(121); montage(extract_patch(A(:,keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15]);
        title('Kept Components');
    subplot(122); montage(extract_patch(A(:,~keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15])
        title('Discarded Components');
        
%% merge found components
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);

%%
display_merging = 1; % flag for displaying merging example
if and(display_merging, ~isempty(merged_ROIs))
    i = 1; %randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
    figure;
    set(gcf,'Position',[300,300,(ln+2)*300,300]);
    for j = 1:ln
        subplot(1,ln+2,j); imagesc(reshape(A_keep(:,merged_ROIs{i}(j)),d1,d2)); 
            title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
    end
    subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
            title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
    subplot(1,ln+2,ln+2);
        plot(1:T,(diag(max(C_keep(merged_ROIs{i},:),[],2))\C_keep(merged_ROIs{i},:))'); 
        hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
        title('Temporal Components','fontsize',16,'fontweight','bold')
    drawnow;
end

%% refine estimates excluding rejected components

Pm.p = p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


%% extract dF/F, including "inferred_traces", "filtered_traces" and "raw_traces" 

[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,Df] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values, and background (optional)

extractControl = struct([]); 
[Inferred_traces, Filtered_traces, Raw_traces] = signalExtraction(Yr,A_or,C_or,b2,f2,options.d1,options.d2,extractControl);

%% generate figures 

%-------------------------- Fig 1 ----------------------------------------%
% figure for correlation image, and components 
ln_cl = 'r';
max_number = size(A_or,2); 
ind_show = 1:max_number; 
hf = figure; 
ha = axes('Parent',hf); 
clrmap = flipud(cbrewer('seq', 'YlGnBu', 100,'pchip'));
[Coor] = contour_plot_simple(A_or,Cn,options,true,ln_cl,clrmap,ha);  
if exist('matfigsavepath','dir')
    savefig(hf,[matfigsavepath,'processed_im_rois.fig']); 
else
    warning('Directory does not exist!! Skipping saving file');
end


%-------------------------- Fig 2 ----------------------------------------%
% figure of dF/F for each temporal component 

sx = min([options.sx,floor(d1/2),floor(d2/2)]);
int_x = zeros(max_number,2*sx);
int_y = zeros(max_number,2*sx);
cm = com(A_or,d1,d2);
imfiles = cell(1,max_number); 
for i=1:max_number
    Atemp = reshape(A_or(:,i),d1,d2);
    int_x(i,:) = round(cm(i,1)) + (-(sx-1):sx);
    if int_x(i,1)<1
        int_x(i,:) = int_x(i,:) + 1 - int_x(i,1);
    end
    if int_x(i,end)>d1
        int_x(i,:) = int_x(i,:) - (int_x(i,end)-d1);
    end
    int_y(i,:) = round(cm(i,2)) + (-(sx-1):sx);
    if int_y(i,1)<1
        int_y(i,:) = int_y(i,:) + 1 - int_y(i,1);
    end
    if int_y(i,end)>d2
        int_y(i,:) = int_y(i,:) - (int_y(i,end)-d2);
    end
    Atemp = Atemp(int_x(i,:),int_y(i,:));
    Atemp(1 ,:) = 1;
    Atemp(end ,:) = 1;
    Atemp(:,1) = 1;
    Atemp(:,end) = 1;
    
    imfiles{i} = full(Atemp); 
end

erraccum = zeros(1,max_number); 
for i=1:max_number % determine outliers / false positives 
    yf = hampel(Filtered_traces.dfof(i,:));
    erraccum(i) = norm(yf - Filtered_traces.dfof(i,:)); 
end
fpos = find(erraccum>2); % 2 is the SNR threshold 
tpos = find(erraccum<=2);
Ttrace = (1:size(C_or,2)).*(1./fps); % sec 
tracemat_maxnorm = zeros(max_number,length(Ttrace)); 
tracemat_maxnorm(tpos,:) = Filtered_traces.dfof(tpos,:);
tracemat_maxnorm(tpos,:) = tracemat_maxnorm(tpos,:)./repmat(max(tracemat_maxnorm(tpos,:),[],2),1,length(Ttrace));
tracemat_maxnorm(fpos,:) = zeros(length(fpos),length(Ttrace)); 


hf = figure; clx_trace = brewermap(round(max_number*1.5),'Dark2'); 
%set(hf,'position',[197 55  1080  750]);
nc = ceil(4*sqrt(max_number/12)); 
nr = ceil(max_number/nc); 
ha = subaxis(4,4, [1:8],'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05); 
contour_plot_simple(A_or,Cn,options,true,ln_cl,clrmap,ha);  
title('Correlation image'); 
subaxis(4,4, [9 10 13 14] ,'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05); 
montage(imfiles,'DisplayRange', [],'size',[nr nc]);
colormap(clrmap); 
title('Spatial footprint')
subaxis(4,4, [11 12 15 16] ,'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05); 
for i=1:max_number
    plot(Ttrace,tracemat_maxnorm(i,:) + (i-1) ,'-','color',clx_trace(i,:),'linewidth',1); hold on; 
    plot(Ttrace,repmat((i-1),1,length(Ttrace)) ,'--k','linewidth',0.5); hold on; 
end
set(gca,'ylim',[0 max_number]); box off; 
ylabel('Components'); 
xlabel('Time (sec)');
get(ha,'TightInset');
title('Filtered traces: dF/F0');

if exist('matfigsavepath','dir')
    savefig(hf,[matfigsavepath,'rois_filtered_traces.fig']); 
else
    warning('Directory does not exist!! Skipping saving file');
end



%% display components

options.save_avi = 0; 
plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

%% make movie
if (1)  
    make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)
end

%% run movie 

% run_movie.m

%% Show distance versus dF/F0 correlation relationship 

dataum = Filtered_traces.dfof';
ccoef = corrcoef(dataum); 
cm = com(A_or,d1,d2); 
dd = pdist2(cm,cm); 
dd_cm = []; 

for iis=1:max_number
    for jjs=1:max_number
        if jjs>iis
            dd_cm = [dd_cm; [dd(iis,jjs) ccoef(iis,jjs)]]; 
        end
    end
end

[~,idx] = sort(dd_cm(:,1)); % sort just the first column
dd_cm = dd_cm(idx,:);

FS = 15;
figure; set(gcf,'position',[147  317  1204  478]);
subplot(1,3,1); 
imagesc(dd); colormap gray; colorbar; 
xlabel('Component #'); ylabel('Component #'); 
title('Pairwise distance (a.u.)'); 
set(gca,'fontsize',FS); axis square; axis tight;
subplot(1,3,2); 
imagesc(ccoef); colormap gray; colorbar; 
xlabel('Component #'); ylabel('Component #'); 
title('Correlation coefficient of dF/F0'); 
set(gca,'fontsize',FS); axis square; axis tight;
subplot(1,3,3); 
plot(dd_cm(:,1),dd_cm(:,2),'ok','markerfacecolor','k','markersize',4);
xlabel('Pairwise distance (a.u.)'); 
ylabel('Correlation coefficient of dF/F0'); axis square; axis tight;
set(gca,'fontsize',FS);

if exist('matfigsavepath','dir')
    savefig(hf,[matfigsavepath,'respcorr_vs_pairwise_dist.fig']); 
else
    warning('Directory does not exist!! Skipping saving file');
end



