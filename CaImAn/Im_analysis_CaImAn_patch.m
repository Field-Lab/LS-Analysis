 
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
datanam = 'Data_1200001'; 
datadat = '2020-06-25-0';
whicheye = 'RE';
tiffpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/',char(datanam),'/'];
fijiprojim = [];
if ~exist([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs'],'dir')
    mkdir([initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs']);
end
matfigsavepath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/',char(datadat),'/',char(whicheye),'/Matlab_outputs/'];

% Do you want to save figs, or run in debug mode? 
save_fig = false; 

% Names of videos 
vid_fullim_roi_fluorescentTrace = [matfigsavepath,'vid_',char(datanam),'_fullim_roi_fluorescentTrace.mp4'];

% Do you have stimulus? 
stim_exist = true;
stim.waveform = 'pulse'; 
stim.period = 4; % sec 
stim.amplitude = 1; % sec 
stim.pulse_width = 25/1000; % sec  



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
%   (2) Remove redundant regions of the image (Suggested: select this region after
%       watching the activity in FIJI)
%   (3) Denoise images using either pre-trained deep learning network
%   (4) Identify sequence of images where light was flashed (not implemented yet)

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
decaytconst = 270/1000; % sec (GCaMP7f: from Dana etal. 2018)

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
range_med = [double((median(medZim(:))-quantile(medZim(:),1-qtl))<=min(medZim(:)))*min(medZim(:)) + ...
    double( (median(medZim(:))-quantile(medZim(:),1-qtl))>min(medZim(:)) )*(median(medZim(:))-quantile(medZim(:),1-qtl)) ...
    double( (median(medZim(:))+quantile(medZim(:),qtl))>=max(medZim(:)) )*max(medZim(:)) + ...
    double( (median(medZim(:))+quantile(medZim(:),qtl))<max(medZim(:)) )*(median(medZim(:))+quantile(medZim(:),qtl))];
range_std = [double((median(stdZim(:))-quantile(stdZim(:),1-qtl))<=min(stdZim(:)))*min(stdZim(:)) + ...
    double( (median(stdZim(:))-quantile(stdZim(:),1-qtl))>min(stdZim(:)) )*(median(stdZim(:))-quantile(stdZim(:),1-qtl)) ...
    double( (median(stdZim(:))+quantile(stdZim(:),qtl))>=max(stdZim(:)) )*max(stdZim(:)) + ...
    double( (median(stdZim(:))+quantile(stdZim(:),qtl))<max(stdZim(:)) )*(median(stdZim(:))+quantile(stdZim(:),qtl))];
hf1 = fig_pintomonitor(); 
subplot(1,2,1); 
imshow(medZim, range_med); title('Median image'); 
subplot(1,2,2); 
imshow(stdZim, range_std); title('Std image'); 
if save_fig
    savefig(hf1,[matfigsavepath,datanam,'_Median_Std_image.fig']); 
end
close(hf1); 


% Ask user for selecting Area of Interest
hf2 = fig_pintomonitor; 
imshow(medZim,[]); imcontrast; hold on; title('Select the vertices of rectangular Area of Interest'); 
[x,y] = deal([]); 
for i=1:2
    [x(i),y(i)] = ginput(1);
    h = plot(x(i),y(i),'or','markersize',10,'linewidth',3); hold on; 
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
end
x = sort(ceil(x)); y = sort(ceil(y)); 
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

% ------------------------------ Step 3 -----------------------------------
% Extract radius of putative ROIs - cell bodies, dendritic spines etc. 
% This will be used for background brightness adjustment and by CaImAn (as
% the standard deviation of Gaussian kernel for spatial segmentation; std = 
% half the size of neuron)
% Limitation: Current version allows only 1 size of ROIs. 

hf3 = fig_pintomonitor; 
imshow(medZim,[]); hold on; 
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

% Denoising image based on Non-local means filtering 
im_uint16_red_denoised = zeros(size(im_uint16_red),'uint16'); 
parfor_progress(num_images);
parfor k=1:num_images
    im_uint16_red_denoised(:,:,k) = imnlmfilt(im_uint16_red(:,:,k),'SearchWindowSize',15,'ComparisonWindowSize',5);
    parfor_progress;
end
parfor_progress(0);
hf4 = fig_pintomonitor(); 
subplot(1,2,1); imshow(median(im_uint16,3),[]); title('Before denoising: Median projection'); 
subplot(1,2,2); imshow(median(im_uint16_red,3),[]); title('After denoising: Median projection'); 
denoiselogical = questdlg( 'Keep/discard denoised images.','Denoise comparison.',...
    ['Keep'],['Discard'],['Discard']);
if strcmpi(denoiselogical, 'Keep')
    im_uint16_red = im_uint16_red_denoised; 
end
clear im_uint16_red_denoised; 
close(hf4); 
    

% % ------------------------------ Step 7 -----------------------------------
% % % Save processed frames as tiff file (this will help with 'memmap'ing later) 
% processedim_savepath = fullfile([matfigsavepath,datanam,'_processed.tif']);
% if exist(processedim_savepath,'file'); delete(processedim_savepath); end
% imwrite(im_uint16_red(:,:,1),processedim_savepath);
% tic
% for ni=2:100
%     imwrite(im_uint16_red(:,:,ni),processedim_savepath,'WriteMode','append');
% end
% toc
    
% ------------------------------ Step 7 -----------------------------------
% Save processed frames as tiff file (this will help with 'memmap'ing later) 
processedim_savepath = fullfile([matfigsavepath,datanam,'_processed.tif']);
if exist(processedim_savepath,'file'); delete(processedim_savepath); end
if ~isempty(dir(fullfile(matfigsavepath,[datanam,'_processed.mat']))); delete(fullfile(matfigsavepath,[datanam,'_processed.mat'])); end
if exist('tiffObj','var'); close(tiffObj); clear tiffObj; end 
clear tagstruct; 
tiffObj = Tiff(processedim_savepath,'w8'); 
tagstruct.Compression = Tiff.Compression.None;
tagstruct.BitsPerSample = info(1).BitsPerSample;
tagstruct.SamplesPerPixel = info(1).SamplesPerPixel;
tagstruct.SampleFormat = Tiff.SampleFormat.UInt; 
tagstruct.RowsPerStrip = size(im_uint16_red,1); 
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.(info(1).PlanarConfiguration);
tagstruct.ImageLength = size(im_uint16_red,1);
tagstruct.ImageWidth = size(im_uint16_red,2); 
tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
parfor_progress(num_images);
for tf=1:num_images 
    tiffObj.setTag(tagstruct);
    tiffObj.write(im_uint16_red(:, :, tf));
    if tf ~= num_images
       tiffObj.writeDirectory();
    end
    parfor_progress();
end
parfor_progress(0);
tiffObj.close(); 




%% ########################################################################
% ----------------------------- Run CaImAn --------------------------------
% #########################################################################

%% Create object and set parameters

CNM = CNMF();                                    % contruct CNMF object


is_memmaped = true;                              % Assumption: supplied data is not mapped for memory optimization
tsub = 1;                                        % temporal downsampling (default: 1. You can try 5 for 30Hz imaging rate or larger, e.g. 8-10)
batch_size = num_images;                         % read chunks of that size
batch_size = round(batch_size/tsub)*tsub;        % make sure batch_size is divisble by tsub
first_frame = 1;                                 % default first frame = 1; 

if is_memmaped
    CNM.readFile(processedim_savepath,is_memmaped);
else
    CNM.readFile(processedim_savepath,is_memmaped,first_frame,batch_size); % load only a part of the file due to memory
end


K = 8;                                          % number of components to be found
gSig = rad;                                     % std of gaussian kernel (size of neuron)  
overlap = ceil(1.5.*gSig.*[1,1]);                     % amount of overlap in each dimension (optional, default: [4,4])
patch_size = [32,32];                           % size of each patch along each dimension (optional, default: [32,32])
p = 2;                                          % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
gnb = 1;                                        % number of global background components
merge_thr = 0.8;                                % merging threshold



options = CNMFSetParms(...
    'd1',CNM.dims(1),'d2',CNM.dims(2),...
    'search_method','dilate',...                % search locations when updating spatial components
    'deconv_method','constrained_foopsi',...    % neural activity deconvolution method
    'p',p,...                                   % order of calcium dynamics
    'ssub',1,...                                % spatial downsampling when processing (default: 1)
    'tsub',tsub,...                             % further temporal downsampling when processing, useful for long recordings or temporal averaging for fast frame rate 
    'nb',1,...                                  % number of background components (per patch)
    'merge_thr',merge_thr,...                   % merging threshold
    'cluster_pixels',true,'flag_g',true,...     % cluster pixels based on closeness of noise PSD
    'gSig',gSig,... 
    'max_size_thr',300,'min_size_thr',10,...    % max/min acceptable size for each component
    'spatial_method','regularized',...          % method for updating spatial components: 'regularized' or 'constrained'
    'thr_method','nrg',...                      % post-processing spatial components: 'max': discard pixel values<max, 'nrg': boundary with 99% energy explained
    'df_prctile',50,...                         % take the median of background fluorescence to compute baseline fluorescence 
    'fr',fps,...                                % imaging frame rate (Hz): (default: 30)
    'space_thresh',0.5,...                      % space correlation acceptance threshold
    ...
    'min_SNR',2,...                             % trace SNR (unit of norm std) acceptance threshold. Used in calculating threshold in compute_event_exceptionality.
    ...
    'cnn_thr',0.2,...                           % cnn classifier acceptance threshold
    'skip_frame', 1,...                         % skip frames when showing the video (default: 1 (no skipping))
    'make_avi', 1,...                           % flag for saving avi video (default: 0)
    'show_background',1,...                     % flag for displaying the background in the denoised panel (default: 1)
    'show_contours',0,...                       % flag for showing the contour plots of the patches in the FoV (default: 0)
    'decay_time',decaytconst,...                % decay constant of GCaMP signal 
    'sx',10,...                                 % half size of representative patches (default: 16)  - for plotting 
    'ind',[1:10],...                            % choose some example components for saving (or creating movies)
    'name',vid_fullim_roi_fluorescentTrace,...  % name of movie file   
    'create_memmap', true);
    %'split_data',true,...                      % reduce memory overhead 

    
CNM.optionsSet(options);
CNM.gnb = gnb;
CNM.K = K;
CNM.patch_size = patch_size;                    % size of each patch along each dimension (optional, default: [32,32])
CNM.overlap = overlap;                          % amount of overlap in each dimension (optional, default: [4,4])


%% Create and fit all patches

CNM.createPatches();                            % create patches
CNM.fitPatches();                               % fit patches 


%% component classification

CNM.evaluateComponents();   % evaluate spatial components based on their correlation with the data
CNM.CNNClassifier('')       % evaluate spatial components with the CNN classifier
CNM.eventExceptionality();  % evaluate traces
CNM.keepComponents();       % keep the components that are above certain thresholds

%% merge components 

CNM.merge();                % merge 
CNM.displayMerging();


%% Update temporal and spatial components after classification by CNN  

CNM.updateSpatial();
CNM.updateTemporal();
CNM.extractDFF();            % extract detrended DF/F values 


%% Get correlation image

CNM.correlationImage();      % compute max of the batchwise correlation images 

%% Generate plots

CNM.plotComponentsGUI();     % display all components
hf6 = fig_pintomonitor(gcf);


%-------------------------- Fig 1 ----------------------------------------%
% figure for correlation image, and components 

ln_cl = 'r';
clrmap = (brewermap(100,'PRGn')); 
hf7 = fig_pintomonitor; 
ha6 = axes('Parent',hf7); 
[Coor] = contour_plot_simple(CNM.A,CNM.CI,CNM.options,true,ln_cl,clrmap,ha6);  
% if exist('matfigsavepath','dir')
%     savefig(hf,[matfigsavepath,datanam,'_processed_im_rois.fig']); 
% else
%     warning('Directory does not exist!! Skipping saving file');
% end


%-------------------------- Fig 2 ----------------------------------------%
% figure of spatial components 
d1 = CNM.options.d1; d2 = CNM.options.d2;
numcomp = size(CNM.A,2); 
sx = min([options.sx,floor(d1/2),floor(d2/2)]);
int_x = zeros(numcomp,2*sx);
int_y = zeros(numcomp,2*sx);
cm = com(CNM.A,d1,d2);
imfiles = cell(1,numcomp); 
for i=1:numcomp
    Atemp = reshape(CNM.A(:,i),d1,d2);
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
nc = ceil(4*sqrt(numcomp/12)); 
nr = ceil(numcomp/nc);
clrmap = flipud((brewermap(100,'Greys')));

 
hf8 = fig_pintomonitor; 
montage(imfiles,'DisplayRange', [],'size',[nr nc]);
colormap(clrmap); 
title(sprintf('Spatial footprints of %d components.',numcomp));
% savefig(hf8,[matfigsavepath,datanam,'_spatial_footprints.fig']); 

%-------------------------- Fig 3 ----------------------------------------%
% figure of dF/F for each temporal component 
clx_trace = brewermap(round(size(CNM.C_df,1)*1.5),'Dark2'); 
Ttrace = (1:size(CNM.C_df,2)).*(1./fps); % sec 
hf9 = fig_pintomonitor; 
for i=1:size(CNM.C_df,1)
    plot(Ttrace,CNM.C_df(i,:) + (i-1) ,'-','color',clx_trace(i,:),'linewidth',1); hold on; 
    plot(Ttrace,repmat((i-1),1,length(Ttrace)) ,'--k','linewidth',0.5); hold on; 
end
set(gca,'ylim',[0 size(CNM.C_df,1)]); box off; 
ylabel('Components'); 
xlabel('Time (sec)');
title('Detrended traces: dF/F0');
% savefig(hf9,[matfigsavepath,datanam,'_detrended_dF_F0.fig']); 


%-------------------------- Fig 4 ----------------------------------------%
% figure of stimulus and trial repeated response 
framedur_s = 1/fps; % ms
total_time_s = num_images * framedur_s; % s 
time_s = framedur_s .* (1:num_images) ; % s 
stimstartid = (find(mod(time_s, stim.period)==0))+1; 
if length(stimstartid)==ceil(time_s(end)/stim.period) && stimstartid(end)>num_images
    stimstartid = [1 stimstartid(1:end-1)];
end
numfr_pertrial = round(mean(diff(stimstartid))); 

hf10 = fig_pintomonitor;
for numcomp=1:size(CNM.C_df,1)
    yfull = CNM.C_df(numcomp,:); 
    outlierid = find(yfull > 10.*std(yfull)); 
    yfull(outlierid) = mean(yfull); 
    tbin =  [1:numfr_pertrial].*framedur_s; 
    
    for i=1:length(stimstartid)
        plot(tbin, yfull((i-1)*numfr_pertrial+1:i*numfr_pertrial),'-k'); hold on; 
    end
    title(sprintf('Component %d',numcomp)); 
    pause; 
    cla;
end










