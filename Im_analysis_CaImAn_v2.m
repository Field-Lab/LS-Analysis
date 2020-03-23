 
%% Set up data path 
caimanpath = 'Z:/lab/Experiments/Array/Shared/sroy/CaImAn-MATLAB-master/';
tiffpath = ['Z:/lab/Experiments/Imaging/Light_sheet/Analysis/2020-03-18-0/',...
    'RE_multipagetiffs/RGC_somas_ShadowSide_Custom_bin2_EO_5x_IO_20x_0.4NA_LP_28mW_Exp_50ms_LaserEvoked_1/',...
    'FIJI_Processed/RGC_somas_ShadowSide_Custom_bin2_EO_5x_IO_20x_0.4NA_LP_28mW_Exp_50ms_Laser_1.tif']; 
hier = find(tiffpath=='/',2,'last'); 
savefigpath = tiffpath(1:hier(1)); 

% caimanpath = 'Z:/lab/Experiments/Array/Shared/sroy/CaImAn-MATLAB-master/';
% tiffpath = 'Z:/lab/Experiments/Imaging/Light_sheet/Analysis/2020-03-10-0/Multipagetiff/Data11/Data11.tif';

% caimanpath = '/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Array/Shared/sroy/CaImAn-MATLAB-master/';
% tiffpath = ['/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Imaging/Light_sheet/Analysis/',...
%     '2020-03-18-0/RE_multipagetiffs/RGC_somas_ShadowSide_Custom_bin2_EO_5x_IO_20x_0.4NA_LP_28mW_Exp_50ms_LaserEvoked_1/',...
%     'FIJI_Processed/RGC_somas_ShadowSide_Custom_bin2_EO_5x_IO_20x_0.4NA_LP_28mW_Exp_50ms_Laser_1.tif']; 


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

%% Read image file, remove outliers, denoise images 

% Steps: 
%   (1) Read image files and properties 
%   (2) Remove redundant regions of the image (Suggested: select this region after
%       watching the activity in FIJI)
%   (3) Denoise images using either pre-trained deep learning network
%   (4) Identify sequence of images where light was flashed (not implemented yet)
%   (5) 


% ------------------------------ Step 1 -----------------------------------
% Read image properties 
info = imfinfo(tiffpath);
num_images = numel(info);
bitdepth = unique([info.BitDepth]); 
dims = [info(1).Height info(2).Width]; 
AOI_red{1} = 1:dims(1); % Area of interest (reduced)
AOI_red{2} = 1:dims(2); 


% ------------------------------ Step 2 -----------------------------------
% Calcium image files can be big, we want to minimize redundancy in the
% image set for analysis as much as possible 

% Max Z projection for identification of Area of Interest 
im_uint16 = zeros([dims num_images],'uint16'); 
parfor k=1:num_images
    im_uint16(:,:,k) = imread(tiffpath,k); 
    fprintf(sprintf('Read %d of %d images\n',k,num_images));
end
medZim = median(im_uint16,3);  % Median projection

% Ask user for selecting Area of Interest
hf = figure(1); imshow(medZim,[]); hold on; title('Select the vertices of rectangular Area of Interest'); 
[x,y] = deal([]); 
for i=1:2
    [x(i),y(i)] = ginput(1);
    h = plot(x(i),y(i),'or','markersize',12,'linewidth',3); hold on; 
    set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
end
x = sort(x); y = sort(y); 
AOI_x = [x(1) x(2) x(2) x(1) x(1)]; % Area of Interest
AOI_y = [y(1) y(1) y(2) y(2) y(1)]; 
plot(AOI_x,AOI_y,'-y','linewidth',6); legend('Area for analysis'); 
saveas(hf, [savefigpath,'AOI_selection'], 'fig'); 

% Generate a smaller set of image 
im_uint16_red = im_uint16(min(AOI_y):max(AOI_y),min(AOI_x):max(AOI_x),:); 
dims_red = [size(im_uint16_red,1) size(im_uint16_red,2)];


% ------------------------------ Step 3 -----------------------------------
% Denoise images using either (1) Total Variation L1 model, or, (2) Matlab's trained CNN classifier 
% ** Need to implement 'parforbar'
% ** Need to set up a GUI for iteratively selecting regularization
% parameter 

% Estimate total time
net = denoisingNetwork('DnCNN');
[tt,nhr] = deal(zeros(1,2)); 
tic; ill_medZim_denoised{1} = TVL1(medZim, 0.5, 100); tt(1) = toc; 
tic; ill_medZim_denoised{2} = denoiseImage(medZim,net); tt(2) = toc; 
for i=1:2
    ttime = ceil(tt(i)*num_images); nhr(i) = (ttime/60/60); 
end

% Denoise images 
answ = questdlg( 'Which method do you want to use for DeNoising images?','De-noise image set.',...
    ['Total Variation Regularization L1 (Estimated time = ',num2str(nhr(1)),' hrs.)'],...
    ['Matlab Neural Net (Estimated time = ',num2str(nhr(2)),' hrs.)'],'None',...
    ['Total Variation Regularization L1 (Estimated time = ',num2str(nhr(1)),' hrs.)']); 
if strcmp(answ(1),'T')
    lambda = 0.5; 
    niter = 100; 
    
    % Need to define appropriate value for regularization parameter: lambda

    im_doub_touse = zeros([dims_red num_images],'double');
    parfor k=1:num_images
        imtemp = TVL1(im_uint16_red(:,:,k), lambda, niter);
        im_doub_touse(:,:,k) = imtemp; 
        fprintf(sprintf('Read %d of %d images\n',k,num_images));
    end
    meth = 'TV-L1';
    clear imtemp; 
elseif strcmp(answ(1),'M')
    im_doub_touse = zeros([dims_red num_images],'double');
    parfor k=1:num_images
        imtemp = cast(im_uint16_red(:,:,k),'double');
        im_doub_touse(:,:,k) = denoiseImage(imtemp,net);
        fprintf('Images %d of %d done\n',k,num_images);
    end
    meth = 'Neural Net'; 
    clear imtemp; 
elseif strcmp(answ(1),'N')
    im_doub_touse = cast(im_uint16_red,'double'); 
    meth = 'No denoising';
end 
save(['Z:/lab/Experiments/Imaging/Light_sheet/Analysis/2020-03-18-0/RE_multipagetiffs/',...
    'RGC_somas_ShadowSide_Custom_bin2_EO_5x_IO_20x_0.4NA_LP_28mW_Exp_50ms_LaserEvoked_1/DeNoisedImageSet.mat'],'im_doub_touse','-v7.3');

hf = figure; set(hf,'position',[403 386 1114 544]);
subplot(121); 
imshow(im_uint16(:,:,900),[]); 
title('Original Image'); 
subplot(122); 
if strcmp(answ(1),'T'); imshow(TVL1(im_uint16_red(:,:,900), 0.5, niter),[]); 
elseif strcmp(answ(1),'M');  imshow(ill_medZim_denoised{2},[]); 
else imshow(ill_medZim,[]); 
end
title('Denoised Image'); 
suptitle(['Denoising. Method : ',char(meth)]); 
saveas(hf, [savefigpath,'deNoising_illustration'], 'fig'); 

% ------------------------------ Step 4 -----------------------------------
% Determine the standard deviation of Gaussian kernel for CaImAn: half the
% size of neuron. We will use a template neuron from median projected image
% to determine the std. 

tempimg = median(Y,3); 
Npts = 6;
hf = figure; imshow(tempimg,[]); hold on; 
title(['You have ',num2str(Npts),' points to define ROI outline']); 
[x_,y_] = deal([]); 
for i=1:Npts
    [x,y] = ginput(1); 
    x_ = [x_ x]; 
    y_ = [y_ y]; 
    if i==Npts
        x_ = [x_ x_(1)];
        y_ = [y_ y_(1)];
    end
    plot(x_,y_,'.-y','markersize',8,'linewidth',1); hold on; 
end
rad = ceil(sqrt(polyarea(x_,y_)/pi)); % fit a circle to a polygon
saveas(hf, [savefigpath,'define_std_GaussKernel'], 'fig'); 



% ------------------------------ Step 5 -----------------------------------

% % Cluster images to identify outliers 
% 
% clustinp = input('Do you want to look for outliers? (y/n) : ','s'); 
% if strcmp(clustinp,'y')
%     zlink = linkage(qtlvals(:,2),'ward'); % clustering on median values 
%     clusters = cluster(zlink,'maxclust',2); % separating outlier group 
%     lcs1 = find(clusters==1); 
%     lcs2 = find(clusters==2); 
% 
%     % Validation figure 
%     nvals = numel(qtlvals(:,2)); 
%     figure; yyaxis left; plot(1:nvals, qtlvals(:,2),'ok','markersize',12); 
%     yyaxis right; plot(lcs1, qtlvals(lcs1,2),'*b',lcs2,qtlvals(lcs2,2),'*m'); 
%     legend('','Cluster1','Cluster2'); 
% 
% 
%     % Choosing a cluster 
%     uinp = input('Which cluster to choose? ','s'); 
%     if strcmp(uinp,'1')
%         imginds = lcs1; 
%     elseif strcmp(uinp,'2')
%         imginds = lcs2;
%     end
% 
%     % Test of contiguity (not yet implemented) 
%     imginds_ = sort(imginds); 
%     if length(imginds_(1):imginds_(end)) == length(imginds_)
%         fprintf('Image group is contiguous! \n'); 
%     else
%         fprintf('Image group is NOT contiguous! \n'); 
%     end
% else
%     imginds = 1:num_images; 
% end


% ------------------------------ Step 3 -----------------------------------
 
   
% % Generate example figure for original and denoised image 
% figure; 
% subplot(1,3,1); 
% imm = im_uint16(:,:,25); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
% imshow(imm,cmapr); clear imm; 
% subplot(1,3,2); 
% imm = denoisedI_uint16(:,:,25); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
% imshow(imm,cmapr);  clear imm;
% subplot(1,3,3); 
% imm = denoisedI_uint16(:,:,25); imm = imm-median(imm(:)); imm = imm-min(imm(:)); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
% imshow(imm,cmapr);  clear imm;

% figure; 
% for i=1:length(imginds)
%     k = imginds(i); 
%     subplot(6,6,i); imm = im_uint16(:,:,k); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
%     imshow(imm, cmapr); 
% end
% figure; 
% for i=1:length(imginds)
%     k = imginds(i); 
%     subplot(6,6,i); imm = denoisedI_uint16(:,:,k); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
%     imshow(imm, cmapr); 
% end

% figure; 
% subplot(1,3,1); [imtemp,map] = imread(nam, 20); imshow(imtemp,map);
% subplot(1,3,2); imshow(denoisedI,[min(denoisedI(:)) max(denoisedI(:))]); 
% subplot(1,3,3); imshow(denoisedI2,[min(denoisedI2(:)) max(denoisedI2(:))]); 

%% Run CaImAn

addpath(caimanpath); 
addpath(genpath([caimanpath,'utilities'])); 
addpath(genpath([caimanpath,'deconvolution'])); 

Y = cast(im_doub_touse,'single'); 
clear im_doub_touse; 
 
%Y = Y - min(Y(:)); 
if ~isa(Y,'single');    Y = single(Y);  end         % convert to single
 
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% Set parameters
 
K = 20;                                           % number of components to be found
tau = rad;                                      % std of gaussian kernel (half size of neuron) 
p = 2; 
 
options = CNMFSetParms('d1',d1,'d2',d2,'p',p,'gSig',tau,'merge_thr',0.80,'nb',2,...
    'min_SNR',3,'space_thresh',0.5,'cnn_thr',0.2);  
% options = CNMFSetParms(...   
%     'd1',d1,'d2',d2,...                         % dimensionality of the FOV
%     'p',p,...                                   % order of AR dynamics    
%     'gSig',tau,...                              % half size of neuron
%     'merge_thr',0.80,...                        % merging threshold  
%     'nb',2,...                                  % number of background components    
%     'min_SNR',3,...                             % minimum SNR threshold
%     'space_thresh',0.5,...                      % space correlation threshold
%     'cnn_thr',0.6...                            % threshold for CNN classifier    
%     );
%% Data pre-processing
 
[P,Y] = preprocess_data(Y,p);
%% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

% display centers of found components
Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)

figure;
imagesc(double(Cn)); colormap gray; 
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

%% classify components

rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        % this test will keep processes
                                        
%% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
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


%% do some plotting

[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

figure;
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
%savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)

%% display components
options.save_avi = 1; 
plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

%% make movie
if (0)  
    make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)
end





