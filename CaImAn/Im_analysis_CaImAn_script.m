%% Set up utils and data path 

% Choose architecture: 
%   1. Linux
%   2. MacOS 
%   3. Windows
carch = computer('arch'); 


% Set path directives 
if regexpi(carch,'glnx')
    initpath = '/Volumes/dusom_fieldlab/All_Staff/lab/';
elseif regexpi(carch,'maci') 
    initpath = '/Volumes/dusom_fieldlab/All_Staff/lab/';
elseif regexpi(carch,'win')
    initpath = 'Z:/lab/';
end
caimanpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/CaImAn/CaImAn-MATLAB-master/'];
normcorrepath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/NoRMCorre-master/'];
addpath(genpath(caimanpath));
addpath(genpath(normcorrepath));
addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/Internal_utils/']));
addpath(genpath([initpath,'Experiments/Imaging/Light_Sheet/Analysis/Matlab/External_utils/']));



% Set full path for raw data, scripts and analyzed data and figures 
datanam = 'Data_1300001'; 
datadat = '2020-07-07-0';
whicheye = 'RE_ventral'; 
tiffpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/',char(datanam),'/'];
fijiprojim = [];
if ~exist([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs'],'dir')
    mkdir([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs']);
end
matfigsavepath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs/'];

% Do you want to save figs, or run in debug mode? 
save_fig = true; 

% Names of videos 
vid_Ca_spatiotemporal_resp = [matfigsavepath,'vid_',char(datanam),'_fullim_roi_fluorescentTrace.mp4'];
make_video = false; 

% Do you have stimulus? 
stim_exist = true;
stim.waveform = 'square'; 
stim.period = 2; % sec 
stim.amplitude = 80; % sec 
stim.pulse_width = []; % sec 
stim.repeat = true; 



%% Set up parallel workers, gpu arrays, located mex files and screen preference 

% Setup parallel workers 
v = ver; 
popt = any(strcmp({v.Name}, 'Parallel Computing Toolbox')); 
if popt 
    pp = gcp('nocreate'); 
    ncores = maxNumCompThreads('automatic'); 
    if ncores > 1 && isempty(pp)
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
%   (2) Select FOV containing the most prominent structures
%   (3) Select size of ROIs (current implementation works for only 1 size (for ex. soma or dendritic spines, not both) 
%   (4) Adjust uneven brightness in images due to shadow or uneven  illumination 
%   (5) Destriping images - specific to light sheet (not implemented yet)
%   (6) Denoise images (sequentially) based on nonlocal means filtering

% ****** Implement sparse/Tall arrays for memory efficiency 

% ------------------------------ Step 1 -----------------------------------
% Read image properties 
tiffdirinfo = dir(fullfile(tiffpath, '*.tif'));  
chunklim = zeros(length(tiffdirinfo),2); 
for fln = 1:length(tiffdirinfo)
    info = imfinfo([tiffpath,tiffdirinfo(fln).name]);
    if fln==1
        chunklim(fln,1:2) = [1 numel(info)];
    else
        chunklim(fln,1:2) = chunklim(fln-1,2)+[1 numel(info)];
    end
end
num_images = chunklim(end,end); 
prompt = {'Total # of images: ','Enter range of images to analyze: '};
dlgtitle = 'Number of images';
dims = [1 50]; % size of dialogue box in pixels 
definput = {num2str(num_images),sprintf('%d - %d',1,num_images)};
answer = inputdlg(prompt,dlgtitle,dims,definput); 
image_rng = cellfun(@str2double,strsplit(answer{2},'-')); 
num_images = diff(image_rng)+1; 
bitdepth = info(1).BitDepth; 
dims = [info(1).Height info(2).Width]; 

strextrct = info(1).ImageDescription(regexpi(info(1).ImageDescription,'Exposure'):regexpi(info(1).ImageDescription,'Exposure')+18); 
fps = 1/str2double(strextrct(regexpi(strextrct,'=')+2:regexpi(strextrct,'=')+8)); % frame rate (/sec)
decaytconst = 270/1000; % sec (GCaMP7f: from Dana et.al 2018)

% ------------------------------ Step 2 -----------------------------------
% Calcium image files can be big, we want to minimize redundancy in the
% image set for analysis as much as possible 

% Max std projection for identification of Area of Interest 
im_uint16 = zeros([dims num_images],'uint16'); 
startim = image_rng(1); 
endim = image_rng(2);  
for fln = 1:length(tiffdirinfo)
    parfor_progress(diff(chunklim(fln,:))+1); 
    fullfilepath = [tiffpath,tiffdirinfo(fln).name]; 
    parfor k=chunklim(fln,1):chunklim(fln,2)
        im_uint16(:,:,k) = imread(fullfilepath, k-chunklim(fln,1)+1); 
        parfor_progress; 
    end
    parfor_progress(0); 
end
im_uint16(:,:,endim+1:end) = []; 
im_uint16(:,:,1:startim-1) = []; 


% Get std and median projections 
medZim = median(im_uint16,3);  % Median projection
stdZim = std(double(im_uint16),0,3); % std projection
qtl = 0.75; % upper quantile value 
range_med = get_intensity_range(medZim); 
range_std = get_intensity_range(stdZim); 
hf1 = fig_pintomonitor(); 
ha11 = subplot(1,2,1); 
imshow(medZim, range_med, 'Parent', ha11); title('Median image'); 
ha12 = subplot(1,2,2); 
imshow(stdZim, range_std, 'Parent', ha12); title('Std image'); 
if save_fig
    savefig(hf1,[matfigsavepath,datanam,'_Median_Std_image.fig']); 
end
close(hf1); 


% Ask user for selecting Area of Interest
hf2 = fig_pintomonitor; 
range_med = get_intensity_range(medZim); 
imshow(medZim, range_med); imcontrast;  hold on; title('Select the vertices of rectangular Area of Interest'); 
[x,y] = deal([]); 
for i=1:2
    [x(i),y(i)] = ginput(1);
    x(i) = round(x(i)); 
    y(i) = round(y(i)); 
    if x(i)<1; x(i)=1; end
    if y(i)<1; y(i)=1; end
    if x(i)>size(medZim,2); x(i)=size(medZim,2); end
    if y(i)>size(medZim,1); y(i)=size(medZim,1); end 
    h = plot(x(i),y(i),'or','markersize',8,'linewidth',3); hold on; 
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
end
AOI_x = [x(1) x(2) x(2) x(1) x(1)]; % Area of Interest
AOI_y = [y(1) y(1) y(2) y(2) y(1)]; 
plot(AOI_x,AOI_y,'-y','linewidth',3); legend('Area for analysis'); 
if save_fig
    savefig(hf2,[matfigsavepath,datanam,'_AOI_selection.fig']); 
end
close(hf2); 


% Generate a smaller set of image 
im_uint16_red = im_uint16(min(AOI_y):max(AOI_y),min(AOI_x):max(AOI_x),:); 
dims_red = [size(im_uint16_red,1) size(im_uint16_red,2)];
medZim = median(im_uint16_red,3); % median projection

% ------------------------------ Step 3 -----------------------------------
% Extract radius of putative ROIs - cell bodies, dendritic spines etc. 
% This will be used for background brightness adjustment and by CaImAn (as
% the standard deviation of Gaussian kernel for spatial segmentation; std = 
% half the size of neuron)
% Limitation: Current version allows only 1 size of ROIs. 

hf3 = fig_pintomonitor; 
range_medZim = get_intensity_range(medZim); 
imshow(medZim,range_medZim); imcontrast; hold on; 
title('Zoom into a region for selecting ROIs, then press Enter!');
zoom(hf3,'on'); 
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
if save_fig
    savefig(hf3,[matfigsavepath,datanam,'_define_std_GaussKernel.fig']); 
end
close(hf3); 



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
% De-striping images (stripes from light-sheet beam entry) : To be added !
% Method 1 : FFT and inverseFFT of mask
% Method 2 : Multidirectional Stripe Remover based on Nonsubsampled
% Method 3: Contour Wavelet Transform 


% ------------------------------ Step 6 -----------------------------------
% Denoise images using either (1) Non-local means filtering, or, (2) CNN classifier 

denoiseim = questdlg( 'Do you want to denoise images?','De-noising images.',...
    ['Yes'],['No'],['No']);
if strcmpi(denoiseim,'Yes')
    im_uint16_red_denoised = zeros(size(im_uint16_red),'uint16'); 
    parfor_progress(num_images);
    parfor k=1:num_images
        im_uint16_red_denoised(:,:,k) = imnlmfilt(im_uint16_red(:,:,k),'SearchWindowSize',15,'ComparisonWindowSize',5);
        parfor_progress;
    end
    parfor_progress(0);
    medZim_red = median(im_uint16_red_denoised,3);     
    range_med = get_intensity_range(medZim); 
    range_med_red = get_intensity_range(medZim_red); 

    hf4 = fig_pintomonitor(); 
    subplot(1,2,1); imshow(medZim,range_med); title('Before denoising: Median projection'); 
    subplot(1,2,2); imshow(medZim_red,range_med_red); title('After denoising: Median projection');

    denoiselogical = questdlg( 'Keep/discard denoised images.','Denoise comparison.',...
        ['Keep'],['Discard'],['Discard']);
    if strcmpi(denoiselogical, 'Keep')
        im_uint16_red = im_uint16_red_denoised; 
    end
    clear im_uint16_red_denoised; 
    close(hf4); 
end
    


%% ########################################################################
% ----------------------------- Run CaImAn --------------------------------
% #########################################################################


% Create object and set parameters
Y = im_uint16_red; 
if ~isa(Y,'single')    
    Y = single(Y);  
end         
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;   


%overlap = 2.*gSig.*[1,1];% amount of overlap in each dimension (optional, default: [4,4])
%patch_size = [32,32];    % size of each patch along each dimension (optional, default: [32,32])

K = 8;                                              % number of components to be found
gSig = ceil(1.*rad);                                % std of gaussian kernel (half size of neuron) : Need to be Integer 
p = 1;                                              % order of AR model dynamics (default: 1)
pixels = 1:numel(Y)/size(Y,ndims(Y));               % pixels to include when computing the AR coefs
refine_components = false;                          % flag for manual refinement

options = CNMFSetParms(...
    'd1',d1,'d2',d2,...                             % dimensionality of the FOV        
    'p',p,...                                       % order of AR dynamics    
    'gSig',gSig,...                                 % half size of neurons
    'cluster_pixels',true,...                       % cluster pixels based on closeness of noise PSD
    'flag_g',true,...                               % compute global AR coefficients (default: false)
    'pixels',pixels,...                             % pixels to include when computing the AR coefs
    'split_data',true,...                           % reduce memory overhead 
    'merge_thr',0.80,...                            % merge of activity correlation is > 
    'nb',1,...                                      % number of background components (default: 1)  
    'min_SNR',1.5,...                               % minimum SNR threshold
    'space_thresh',0.5,...                          % space correlation threshold
    'cnn_thr',0.2,...                               % threshold for CNN classifier 
    'fr',fps,....                                   % frame rate of image acquisition 
    'decay_time',decaytconst,...                    % decay constant of GCaMP signal 
    'sx',12,...                                     % half size of representative patches (default: 16)  - for plotting 
    'ind',[1:10],...                                % choose some example components for saving (or creating movies) 
    'skip_frame', 1,...                             % skip frames when showing the video (default: 1 (no skipping))
    'show_background',1,...                         % flag for displaying the background in the denoised panel (default: 1)
    'show_contours',0,...                           % flag for showing the contour plots of the patches in the FoV (default: 0)
    'name',vid_Ca_spatiotemporal_resp);



% -------------------------------------------------------------------------
% Data pre-processing
[P,Y] = preprocess_data(Y,p,options);

% -------------------------------------------------------------------------
% Fast initialization of spatial components using greedyROI and HALS
[Ain,Cin,bin,fin,center] = initialize_components(Y,K,gSig,options,P);  % initialize

% -------------------------------------------------------------------------
% Manually refine components (optional)
if refine_components
    Cn = correlation_image(Y); 
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end
   

% -------------------------------------------------------------------------
% Update spatial and temporal components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);


% -------------------------------------------------------------------------
% Classify components (implementation in Pnevmatikakis et.al) 
rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
          

% -------------------------------------------------------------------------                                      
% Further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end


% -------------------------------------------------------------------------                            
% Event exceptionality for catching strong transients (exceptional events) 
fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);


% -------------------------------------------------------------------------
% Keep select components and discard remaining 
keep = (ind_corr | ind_cnn) ; %& ind_exc;
A_keep = A(:,keep);
C_keep = C(keep,:);


% -------------------------------------------------------------------------
% Extract residual signals for each trace 
if exist('YrA','var')
	R_keep = YrA(keep,:); 
else
	R_keep = compute_residuals(Y, A_keep,b,C_keep,f); 
end


% -------------------------------------------------------------------------
% Merge spatially overlapping components with highly correlated temporal activity 
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);


% -------------------------------------------------------------------------
% Refine estimates excluding rejected components 
Pm.p = options.p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


% -------------------------------------------------------------------------
% Extract detrended dF/F 
[F_dff,F0] = detrend_df_f(A2,b2,C2,f2,R_keep,options); 



% -------------------------------------------------------------------------
% Make video of spatial ROIs, spatial background, ROI intensity modulation and fluorescent traces fig_asp_ratio
fig_asp_ratio = 0.85; 
if make_video
    [stim_xtrace, stim_ytrace] = make_calcium_movie(A2, b2, C2, F_dff, vid_Ca_spatiotemporal_resp, options, stim, make_video, fig_asp_ratio); 
end


% -------------------------------------------------------------------------
% Generate stimulus time trace 
if ~exist('stim_xtrace','var')
    t_axis = (1:num_images).*1/fps; % ms 
    stim_xtrace = []; 
    if isstruct(stim)
        if strcmpi(stim.waveform,'pulse')
            numcycles = floor(t_axis(end)/stim.period); 
            stim_ytrace = repmat([0 1 1 0],1,numcycles).*stim.amplitude; 
            for nc=1:numcycles 
                stim_xtrace = [stim_xtrace [0 0 stim.pulse_width stim.pulse_width]+(nc-1)*stim.period]; 
            end
        elseif strcmpi(stim.waveform,'square')
            numcycles = floor(t_axis(end)/stim.period); 
            stim_ytrace = repmat([0 1 1 0],1,numcycles).*stim.amplitude; 
            for nc=1:numcycles 
                stim_xtrace = [stim_xtrace [0 0 stim.period/2 stim.period/2]+(nc-1)*stim.period]; 
            end      
        elseif strcmpi(stim.waveform,'sine')
            numcycles = floor(t_axis(end)/stim.period);
            stim_ytrace = repmat(sin(2.*pi.*(0:0.01:stim.period-0.01)./stim.period).*stim.amplitude,1,numcycles);
            stim_xtrace = linspace(0,numcycles.*stim.period, length(stim_ytrace)); 
        end
    end
end


%% Figures

% Fig 1: temporal trace, dF/F0 trace, residual trace and background for all components (stacked) 
t_axis = (1:num_images).*1/fps; % ms 
numcomp = size(C2,1); 
clx_trace = brewermap(3,'Set2')';
if stim_exist 
    xll = [0 stim.period.*10]; 
else
    xll = [0 min([t_axis(end) 10])]; 
end
extid = find(t_axis<=xll(2),1,'last'); 

if sum(keep)~=0
    hf101 = fig_pintomonitor('','fracx',0.7); 
    %tiledlayout(2,2); 
    %ax1 = nexttile;
    %ax2 = nexttile;
    %ax3 = nexttile;
    %ax4 = nexttile;
    %hold([ax1, ax2, ax3, ax4],'on');
    ax1 = subplot(2,2,1); 
    ax2 = subplot(2,2,2); 
    ax3 = subplot(2,2,3); 
    ax4 = subplot(2,2,4); 
    hold(ax1,'on'); hold(ax2,'on'); hold(ax3,'on'); hold(ax4,'on');
    
    for i=1:numcomp  
        y1 = C2(i,1:extid); 
        y1 = (y1-min(y1))./max((y1-min(y1))); 
        plot(ax1,t_axis(1:extid),y1 + (i-1) ,'-','color',clx_trace(1,:),'linewidth',1); hold(ax1,'on');
        set(ax1,'xlim',xll); box off; 
        if i==numcomp
            yll = get(ax1,'ylim'); 
            plot(ax1,stim_xtrace, stim_ytrace.*diff(yll)./max(stim_ytrace) + yll(1) , '-','color',[1 0 0 0.5]); 
        end
        title(ax1,'Temporal component (scaled)'); 
        xlabel(ax1,'Time (sec)');
        ylabel(ax1,'Components'); 
        drawnow;
        
        y2 = F_dff(i,1:extid); 
        y2 = (y2-min(y2))./max((y2-min(y2)));
        plot(ax2,t_axis(1:extid),y2 + (i-1) ,'-','color',clx_trace(1,:),'linewidth',1); hold(ax2,'on'); 
        set(ax2,'xlim',xll); box off;
        if i==numcomp
            yll = get(ax2,'ylim'); 
            plot(ax2,stim_xtrace, stim_ytrace.*diff(yll)./max(stim_ytrace) + yll(1) , '-','color',[1 0 0 0.5]);
        end
        title(ax2,'Detrended dF/F0');     
        xlabel(ax2,'Time (sec)');
        ylabel(ax2,'Components');
        drawnow;
        
        y4 = YrA2(i,1:extid);        
        y4 = (y4-min(y4))./max((y4-min(y4)));
        plot(ax4,t_axis(1:extid),y4 + (i-1) ,'-','color',clx_trace(2,:),'linewidth',1); hold(ax4,'on'); 
        set(ax4,'xlim',xll); box off;
        if i==numcomp
            yll = get(ax4,'ylim'); 
            plot(ax4,stim_xtrace, stim_ytrace.*diff(yll)./max(stim_ytrace) + yll(1), '-','color',[1 0 0 0.5]); 
        end
        title(ax4,'Residual trace');     
        xlabel(ax4,'Time (sec)');
        ylabel(ax4,'Components');
        drawnow;
    end
    y3 = f2; 
    plot(ax3,t_axis,y3 ,'-','color',clx_trace(3,:),'linewidth',1); box off; hold(ax3,'on'); 
    yll = get(ax3,'ylim'); 
    plot(ax3,stim_xtrace, stim_ytrace.*diff(yll)./max(stim_ytrace) + yll(1) , '-','color',[1 0 0 0.5]);  
    set(ax3, 'xlim', xll); box off; 
	title(ax3,'Temporal background');     
	xlabel(ax3,'Time (sec)');
	ylabel(ax3,'Intensity (au)');
    
    pause(0.2);     
else 
    warning('No ROIs detected'); 
end
suptitle(sprintf('Stim: %s, amplitude %dmV ',stim.waveform,stim.amplitude)); 
if save_fig
    savefig(hf101,[matfigsavepath,datanam,'_temporal_traces.fig']); 
end
 


% Fig 2: Spatial footprint 
ln_cl = 'r';
txt_cl = 'y'; 
max_number = size(A2,2); 
ind_show = 1:max_number; 
backg = reshape(b2,d1,d2);
hf102 = fig_pintomonitor('','fracx',0.6); 
ha = axes('Parent',hf102); 
clrmap = flipud(brewermap(10,'Greys'));
[Coor] = contour_plot_simple(A2,backg,options,true,ln_cl,txt_cl,clrmap,ha);  
title('Spatial components'); 
if save_fig
    savefig(hf102,[matfigsavepath,datanam,'_spatial_footprint.fig']); 
end


% Fig 3: Plot raster and psth 
if stim.repeat 
    framedur_s = 1/fps; % s
    total_time_s = num_images * framedur_s; % s 
    time_s = framedur_s .* (1:num_images); % s 
    stimstartid = (find(mod(time_s, stim.period)==0))+1; 
    if length(stimstartid)==ceil(time_s(end)/stim.period) && stimstartid(end)>num_images
        stimstartid = [1 stimstartid(1:end-1)];
    end
    numfr_pertrial = round(mean(diff(stimstartid))); 
    num_repeats = length(stimstartid);
    stim_x_1trial = stim_xtrace(1:find(stim_xtrace==stim.period,1,'first')); 
    stim_y_1trial = stim_ytrace(1:find(stim_xtrace==stim.period,1,'first'));
    
    hf103 = fig_pintomonitor('','fracx',0.7); 
    nc = ceil(sqrt(size(C2,1))); 
    nr = ceil(size(C2,1)/nc); 
    clx_ht = [linspace(0,1,num_repeats)' linspace(0,0,num_repeats)' linspace(1,0,num_repeats)']; 
    tcks = [0 num_repeats-1];
    tcklbl = {'1',num2str(num_repeats)};  
    clx_grey = 0.2.*[1 1 1];
    [xtrace_accum,ytrace_accum] = deal(zeros(size(C2,1),num_repeats,numfr_pertrial)); 
    for n=1:size(C2,1)
        yfull = F_dff(n,:); 
        outlierid = find(yfull > 10.*std(yfull)); 
        yfull(outlierid) = mean(yfull); 
        tbin = (1:numfr_pertrial).*framedur_s; 
        
        ax = subplot(nr,nc,n); box off; 
        for i=1:num_repeats
            ytrace = yfull((i-1)*numfr_pertrial+1:i*numfr_pertrial); 
            ytrace = ytrace  - min(ytrace); % reset baseline if there is drift 
            plot(tbin, ytrace, '-','color',clx_ht(i,:)); hold on; 
            ytrace_accum(n,i,:) = ytrace; 
            xtrace_accum(n,i,:) = tbin;
        end
        title(sprintf('ROI %d',n)); 
        ylabel('dF/F0'); 
        xlabel('Time (s)');
        set(ax,'FontSize',12,'box','off');
        colormap(ax,clx_ht);
        cb = colorbar;
        cb.Label.String = 'Trial #';
        cb.Ticks = [cb.Ticks(1) cb.Ticks(end)];
        cb.TickLabels = {'1',num2str(num_repeats)};
        cb.FontSize = 12;
        
        yll = get(gca,'ylim');
        plot(stim_x_1trial, stim_y_1trial.*diff(yll)./max(stim_y_1trial) + yll(1),'-','color',clx_grey,'linewidth',2); 
    end
    if save_fig
        savefig(hf103,[matfigsavepath,datanam,'_response_per_trial.fig']);
    end
    
    
    hf104 = fig_pintomonitor('','fracx',0.7); 
    nc = ceil(sqrt(size(C2,1))); 
    nr = ceil(size(C2,1)/nc); 
    for n=1:size(C2,1)
        subplot(nr,nc,n);
        xx = squeeze(xtrace_accum(n,1,:)); 
        yy = squeeze(mean(ytrace_accum(n,:,:),2));
        yystd = squeeze(std(ytrace_accum(n,:,:),[],2)); 
        
        shadedErrorBar(xx,yy,yystd,'-m'); hold on;
        yll = get(gca,'ylim');
        plot(stim_x_1trial, stim_y_1trial.*diff(yll)./max(stim_y_1trial),'-','color',clx_grey,'linewidth',2); 
        ylabel('dF/F0'); 
        xlabel('Time (s)');
        title(sprintf('ROI %d',n)); 
        set(ax,'FontSize',12,'box','off');
    end
    if save_fig
        savefig(hf104,[matfigsavepath,datanam,'_meanresponse_trials.fig']);
    end
    
end











