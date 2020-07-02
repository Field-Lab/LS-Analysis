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
tiffpath = [initpath,'Experiments/Imaging/Light_Sheet/Analysis/Neurofinder_dataset/neurofinder.00.00/images/'];

% Do you want to save figs, or run in debug mode? 
save_fig = false; 




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

%% Read image file 

% Load images
files = dir([tiffpath,'*.tiff']);

dims = size(imread(strcat(tiffpath, files(1).name))); 
imgs = zeros(dims(1),dims(2),length(files),'uint16'); 
parfor i = 1:length(files)
    fname = strcat(tiffpath, files(i).name);
	imgs(:,:,i) = imread(fname);
end
[x, y, z] = size(imgs); 
im_uint16_red = imgs;
num_images = size(im_uint16_red,3); 
clear imgs; 

% Load imaging info 
inds = find(tiffpath=='/',2,'last');
recinfo = loadjson([tiffpath(1:inds(1)),'info.json']); 
fps = recinfo.rate_0x2D_hz;

% GCaMP info 
decaytconst = 280/1000; % sec (GCaMP6f: from Dana etal. 2018)

% Extract size of ROIs
hf3 = fig_pintomonitor; 
imshow(median(im_uint16_red,3),[]); hold on; 
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
close;



% %% Read image file, remove outliers, denoise images 
% 
% % Steps: 
% %   (1) Read image files and properties 
% %   (2) Remove redundant regions of the image (Suggested: select this region after
% %       watching the activity in FIJI)
% %   (3) Denoise images using either pre-trained deep learning network
% %   (4) Identify sequence of images where light was flashed (not implemented yet)
% 
% % ****** Implement sparse/Tall arrays for memory efficiency 
% 
% % ------------------------------ Step 1 -----------------------------------
% % Read image properties 
% tiffdirinfo = dir(fullfile(tiffpath, '*.tif'));  
% chunklim = zeros(length(tiffdirinfo),2); 
% for fln = 1:length(tiffdirinfo)
%     info = imfinfo([tiffpath,tiffdirinfo(fln).name]);
%     if fln==1
%         chunklim(fln,1:2) = [1 numel(info)];
%     else
%         chunklim(fln,1:2) = chunklim(fln-1,2)+[1 numel(info)];
%     end
% end
% num_images = chunklim(end,end); 
% prompt = {'Total # of images: ','Enter range of images to analyze: '};
% dlgtitle = 'Number of images';
% dims = [1 50]; % size of dialogue box in pixels 
% definput = {num2str(num_images),sprintf('%d - %d',1,num_images)};
% answer = inputdlg(prompt,dlgtitle,dims,definput); 
% image_rng = cellfun(@str2double,strsplit(answer{2},'-')); 
% num_images = diff(image_rng)+1; 
% bitdepth = info(1).BitDepth; 
% dims = [info(1).Height info(2).Width]; 
% 
% strextrct = info(1).ImageDescription(regexpi(info(1).ImageDescription,'Exposure'):regexpi(info(1).ImageDescription,'Exposure')+18); 
% fps = 1/str2double(strextrct(regexpi(strextrct,'=')+2:regexpi(strextrct,'=')+8)); % frame rate (/sec)
% decaytconst = 270/1000; % sec (GCaMP7f: from Dana etal. 2018)
% 
% % ------------------------------ Step 2 -----------------------------------
% % Calcium image files can be big, we want to minimize redundancy in the
% % image set for analysis as much as possible 
% 
% % Max std projection for identification of Area of Interest 
% im_uint16 = zeros([dims num_images],'uint16'); 
% startim = image_rng(1); 
% endim = image_rng(2);  
% for fln = 1:length(tiffdirinfo)
%     parfor_progress(diff(chunklim(fln,:))+1); 
%     fullfilepath = [tiffpath,tiffdirinfo(fln).name]; 
%     parfor k=chunklim(fln,1):chunklim(fln,2)
%         im_uint16(:,:,k) = imread(fullfilepath, k-chunklim(fln,1)+1); 
%         parfor_progress; 
%     end
%     parfor_progress(0); 
% end
% im_uint16(:,:,endim+1:end) = []; 
% im_uint16(:,:,1:startim-1) = []; 
% 
% 
% % Get std and median projections 
% medZim = median(im_uint16,3);  % Median projection
% stdZim = std(double(im_uint16),0,3); % std projection
% qtl = 0.75; % upper quantile value 
% range_med = [double((median(medZim(:))-quantile(medZim(:),1-qtl))<=min(medZim(:)))*min(medZim(:)) + ...
%     double( (median(medZim(:))-quantile(medZim(:),1-qtl))>min(medZim(:)) )*(median(medZim(:))-quantile(medZim(:),1-qtl)) ...
%     double( (median(medZim(:))+quantile(medZim(:),qtl))>=max(medZim(:)) )*max(medZim(:)) + ...
%     double( (median(medZim(:))+quantile(medZim(:),qtl))<max(medZim(:)) )*(median(medZim(:))+quantile(medZim(:),qtl))];
% range_std = [double((median(stdZim(:))-quantile(stdZim(:),1-qtl))<=min(stdZim(:)))*min(stdZim(:)) + ...
%     double( (median(stdZim(:))-quantile(stdZim(:),1-qtl))>min(stdZim(:)) )*(median(stdZim(:))-quantile(stdZim(:),1-qtl)) ...
%     double( (median(stdZim(:))+quantile(stdZim(:),qtl))>=max(stdZim(:)) )*max(stdZim(:)) + ...
%     double( (median(stdZim(:))+quantile(stdZim(:),qtl))<max(stdZim(:)) )*(median(stdZim(:))+quantile(stdZim(:),qtl))];
% hf1 = fig_pintomonitor(); 
% ha11 = subplot(1,2,1); 
% imshow(medZim, range_med, 'Parent', ha11); title('Median image'); 
% ha12 = subplot(1,2,2); 
% imshow(stdZim, range_std, 'Parent', ha12); title('Std image'); 
% if save_fig
%     savefig(hf1,[matfigsavepath,datanam,'_Median_Std_image.fig']); 
% end
% 
% 
% % Ask user for selecting Area of Interest
% hf2 = fig_pintomonitor; 
% imshow(medZim,[]); imcontrast; hold on; title('Select the vertices of rectangular Area of Interest'); 
% [x,y] = deal([]); 
% for i=1:2
%     [x(i),y(i)] = ginput(1);
%     h = plot(x(i),y(i),'or','markersize',10,'linewidth',3); hold on; 
%     set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
% end
% x = sort(ceil(x)); y = sort(ceil(y)); 
% AOI_x = [x(1) x(2) x(2) x(1) x(1)]; % Area of Interest
% AOI_y = [y(1) y(1) y(2) y(2) y(1)]; 
% plot(AOI_x,AOI_y,'-y','linewidth',3); legend('Area for analysis'); 
% if save_fig
%     savefig(hf2,[matfigsavepath,datanam,'_AOI_selection.fig']); 
% end
% 
% 
% % Generate a smaller set of image 
% im_uint16_red = im_uint16(min(AOI_y):max(AOI_y),min(AOI_x):max(AOI_x),:); 
% dims_red = [size(im_uint16_red,1) size(im_uint16_red,2)];
% 
% % ------------------------------ Step 3 -----------------------------------
% % Extract radius of putative ROIs - cell bodies, dendritic spines etc. 
% % This will be used for background brightness adjustment and by CaImAn (as
% % the standard deviation of Gaussian kernel for spatial segmentation; std = 
% % half the size of neuron)
% % Limitation: Current version allows only 1 size of ROIs. 
% 
% hf3 = fig_pintomonitor; 
% imshow(medZim,[]); hold on; 
% title('Zoom into a region for selecting ROIs, then press Enter!');
% zoom(hf3,'on'); 
% pause(); 
% title('Select the outline of an ROI using left click, and press Enter when done!');
% zoom reset;
% clear x y; 
% [x_,y_] = deal([]); 
% but = 1; 
% while ~isempty(but)
%     [x,y,but] = ginput(1);
%     x_ = [x_ x];
%     y_ = [y_ y];
%     plot(x_,y_,'.-y','markersize',8,'linewidth',1); hold on;
% end
% title('Template ROI is now selected'); 
% pshape = polyshape(x_,y_,'simplify',true);
% rad = round(sqrt(pshape.area/pi)); % fit a circle to a polygon
% if save_fig
%     savefig(hf3,[matfigsavepath,datanam,'_define_std_GaussKernel.fig']); 
% end
% 
% 
% 
% % ------------------------------ Step 4 -----------------------------------
% % Adjust uneven brightness from stimulus if captured on images 
% if stim_exist
%     stimremv = questdlg( 'Do you want auto-removal of stimulus brightness from images?','Auto-remove stimulus.',...
%     ['Yes'],['No'],['No']);
%     if strcmpi(stimremv,'Yes')
%         se = strel('disk',rad);
%         parfor_progress(size(im_uint16_red,3));
%         im_uint16_red_lumadj = uint16(zeros(size(im_uint16_red)));
%         parfor ki= 1:size(im_uint16_red,3)
%             im_uint16_red_lumadj(:,:,ki) = imtophat(im_uint16_red(:,:,ki), se);
%             parfor_progress;
%         end
%         parfor_progress(0);
%         im_uint16_red = im_uint16_red_lumadj;
%         medZim = median(im_uint16_red,3); 
%         clear im_uint16_red_lumadj; 
%     end 
% end
% 
% % ------------------------------ Step 5 -----------------------------------
% % De-striping images (stripes from light-sheet beam entry) : To be added !
% % Method 1 : FFT and inverseFFT of mask
% % Method 2 : Multidirectional Stripe Remover based on Nonsubsampled
% % Method 3: Contour Wavelet Transform 
% 
% 
% % ------------------------------ Step 6 -----------------------------------
% % Denoise images using either (1) Non-local means filtering, or, (2) CNN classifier 
% 
% % Denoising image based on Non-local means filtering 
% im_uint16_red_denoised = zeros(size(im_uint16_red),'uint16'); 
% parfor_progress(num_images);
% parfor k=1:num_images
%     im_uint16_red_denoised(:,:,k) = imnlmfilt(im_uint16_red(:,:,k),'SearchWindowSize',15,'ComparisonWindowSize',5);
%     parfor_progress;
% end
% parfor_progress(0);
% hf4 = fig_pintomonitor(); 
% subplot(1,2,1); imshow(median(im_uint16,3),[]); title('Before denoising: Median projection'); 
% subplot(1,2,2); imshow(median(im_uint16_red,3),[]); title('After denoising: Median projection'); 
% denoiselogical = questdlg( 'Keep/discard denoised images.','Denoise comparison.',...
%     ['Keep'],['Discard'],['Discard']);
% if strcmpi(denoiselogical, 'Keep')
%     im_uint16_red = im_uint16_red_denoised; 
% end
% clear im_uint16_red_denoised; 
%     
% 
% % % ------------------------------ Step 7 -----------------------------------
% % % % Save processed frames as tiff file (this will help with 'memmap'ing later) 
% % processedim_savepath = fullfile([matfigsavepath,datanam,'_processed.tif']);
% % if exist(processedim_savepath,'file'); delete(processedim_savepath); end
% % imwrite(im_uint16_red(:,:,1),processedim_savepath);
% % tic
% % for ni=2:100
% %     imwrite(im_uint16_red(:,:,ni),processedim_savepath,'WriteMode','append');
% % end
% % toc
%     
% % % ------------------------------ Step 7 -----------------------------------
% % % Save processed frames as tiff file (this will help with 'memmap'ing later) 
% % processedim_savepath = fullfile([matfigsavepath,datanam,'_processed.tif']);
% % if exist(processedim_savepath,'file'); delete(processedim_savepath); end
% % if exist('tiffObj','var'); close(tiffObj); clear tiffObj; end 
% % clear tagstruct; 
% % tiffObj = Tiff(processedim_savepath,'w8'); 
% % tagstruct.Compression = Tiff.Compression.None;
% % tagstruct.BitsPerSample = info(1).BitsPerSample;
% % tagstruct.SamplesPerPixel = info(1).SamplesPerPixel;
% % tagstruct.SampleFormat = Tiff.SampleFormat.UInt; 
% % tagstruct.RowsPerStrip = size(im_uint16_red,1); 
% % tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.(info(1).PlanarConfiguration);
% % tagstruct.ImageLength = size(im_uint16_red,1);
% % tagstruct.ImageWidth = size(im_uint16_red,2); 
% % tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
% % parfor_progress(num_images);
% % for tf=1:num_images 
% %     tiffObj.setTag(tagstruct);
% %     tiffObj.write(im_uint16_red(:, :, tf));
% %     if tf ~= num_images
% %        tiffObj.writeDirectory();
% %     end
% %     parfor_progress();
% % end
% % parfor_progress(0);
% % tiffObj.close(); 


%% ########################################################################
% ----------------------------- Run CaImAn --------------------------------
% #########################################################################

%% Create object and set parameters

Y = im_uint16_red; 
if ~isa(Y,'single')    
    Y = single(Y);  
end        
 
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;   


%% Set parameters


%overlap = 2.*gSig.*[1,1];% amount of overlap in each dimension (optional, default: [4,4])
%patch_size = [32,32];    % size of each patch along each dimension (optional, default: [32,32])


K = 15;                                             % number of components to be found
gSig = rad;                                         % std of gaussian kernel (half size of neuron) : Need to be Integer 
p = 2;                                              
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
    'min_SNR',2,...                                 % minimum SNR threshold
    'space_thresh',0.5,...                          % space correlation threshold
    'cnn_thr',0.2,...                               % threshold for CNN classifier 
    'fr',fps,....                                   % frame rate of image acquisition 
    'decay_time',decaytconst,...                    % decay constant of GCaMP signal 
    'sx',12,...                                     % half size of representative patches (default: 16)  - for plotting 
    'ind',[1:10],...                                % choose some example components for saving (or creating movies) 
    'skip_frame', 1,...                             % skip frames when showing the video (default: 1 (no skipping))
    'make_avi', 0,...                               % flag for saving avi video (default: 0)
    'show_background',1,...                         % flag for displaying the background in the denoised panel (default: 1)
    'show_contours',0);                             % flag for showing the contour plots of the patches in the FoV (default: 0)



%% Data pre-processing
 
[P,Y] = preprocess_data(Y,p,options);

%% Fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,gSig,options,P);  % initialize

%% Manually refine components (optional)

if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end
    

%% Update spatial and temporal components

Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

%% Classify components (implementation in Pnevmatikakis et.al) 

rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        
                                        
%% Further classification with cnn_classifier

try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end     
                            
%% Event exceptionality for catching strong transients (exceptional events) 

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% Keep select components and discard remaining 

keep = (ind_corr | ind_cnn) & ind_exc;

A_keep = A(:,keep);
C_keep = C(keep,:);


%% Extract residual signals for each trace 
if exist('YrA','var')
	R_keep = YrA(keep,:); 
else
	R_keep = compute_residuals(Y, A_keep,b,C_keep,f); 
end

%% Merge spatially overlapping components with highly correlated temporal activity 

[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);

%% Refine estimates excluding rejected components

Pm.p = options.p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


%% Extract detrended dF/F 

[F_dff,F0] = detrend_df_f(A2,b2,C2,f2,YrA2,options); 


[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,Df] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values, and background (optional)

extractControl = struct([]); 
[Inferred_traces, Filtered_traces, Raw_traces] = signalExtraction(Yr,A_or,C_or,b2,f2,options.d1,options.d2,extractControl);


%% Figures

% Fig 1: dF/F0 traces for all components stacked  
t_axis = (1:num_images).*1/fps; % ms 
numcomp = size(C2,1); 
clx_trace = brewermap(1,'Dark2');
if sum(keep)~=0
    hf = fig_pintomonitor; 
    for i=1:numcomp
        y1 = F_dff(i,:);
        plot(t_axis,y1./max(y1(:)) + (i-1) ,'-','color',clx_trace,'linewidth',1); hold on; 
        set(gca, 'ylim', [0 numcomp+1],'ycolor','w'); box off; 
        pause(0.2); 
    end
    title('Detrended dF/F0');     
    xlabel('Time (sec)');
    ylabel('Components'); 
else 
    warning('No ROIs detected'); 
end

% Fig 2: movie of components

