 
%% Set up data path 
caimanpath = '/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Array/Shared/sroy/CaImAn-MATLAB-master/';
tiffpath = '/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Imaging/Light_sheet/2019-09-05-0/Movies/Data5_multipage_tiff/Data5.tif'; 
savedenoiseddata = '/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Imaging/Light_sheet/Analysis/Matlab/'; 
date_ = '2019-09-05-0';
dataindex = 'Data5'; 

info = imfinfo(tiffpath);
num_images = numel(info);
bitdepth = unique([info.BitDepth]); 
dims = [info(1).Width info(2).Height]; 
dims_red = [400 400];
ROI_red = 401:800; 


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
%   (1) Read image files, cast to double for post-processing 
%   (2) Remove contiguous outliers 
%   (3) Denoise images using pre-trained deep learning network

% ------------------------------ Step 1 -----------------------------------
% Read images (for ex. images are dark) 
im_uint16 = zeros([dims_red num_images],'uint16'); 
parfor k=1:num_images
    [imtemp_uint16, map_uint16] = imread(tiffpath,k); 
    imtemp_double = cast(imtemp_uint16(ROI_red,ROI_red), 'double');
    qtlvals(k,:) = quantile(imtemp_double(:),[0.25 0.5 0.75]); 
    
    im_uint16(:,:,k) = imtemp_uint16(ROI_red,ROI_red); 
end

% [imtemp1,map] = imread(tiffpath,25); 
% imtemp_uint16_dn1 = denoiseImage(imtemp1,net); 
% imtemp1doub = cast(imtemp1,'double'); 
% imtemp_double_dn1 = denoiseImage(imtemp1doub,net);  


% ------------------------------ Step 2 -----------------------------------
% Cluster images to identify outliers 
zlink = linkage(qtlvals(:,2),'ward'); % clustering on median values 
clusters = cluster(zlink,'maxclust',2); % separating outlier group 
lcs1 = find(clusters==1); 
lcs2 = find(clusters==2); 

% Validation figure 
nvals = numel(qtlvals(:,2)); 
figure; yyaxis left; plot(1:nvals, qtlvals(:,2),'ok','markersize',12); 
yyaxis right; plot(lcs1, qtlvals(lcs1,2),'*b',lcs2,qtlvals(lcs2,2),'*m'); 
legend('','Cluster1','Cluster2'); 


% Choosing a cluster 
uinp = input('Which cluster to choose? ','s'); 
if strcmp(uinp,'1')
    imginds = lcs1; 
elseif strcmp(uinp,'2')
    imginds = lcs2;
end

% Test of contiguity (not yet implemented) 
imginds_ = sort(imginds); 
if length(imginds_(1):imginds_(end)) == length(imginds_)
    fprintf('Image group is contiguous! \n'); 
else
    fprintf('Image group is NOT contiguous! \n'); 
end


% ------------------------------ Step 3 -----------------------------------
% Use learned deep neural net to denoise images 
net = denoisingNetwork('DnCNN');
denoisedI_uint16 = zeros([dims_red num_images],'uint16');
denoisedI_doub = zeros([dims_red num_images]);
parfor k=imginds(1):imginds(end)
    imtemp = im_uint16(:,:,k); 
    denoisedI_uint16(:,:,k) = denoiseImage(imtemp,net); 
    denoisedI_doub(:,:,k) = cast(denoisedI_uint16(:,:,k),'double'); 
    fprintf(sprintf('Images %d of %d done\n',k,length(imginds))); 
end

% Select contiguous denoised images (and save) 
denoisedI_uint16_cut = denoisedI_uint16(:,:,imginds); 
denoisedI_doub_cut = denoisedI_doub(:,:,imginds); 
%save([savedenoiseddata,date_,'_',dataindex,'_','denoisedImgs.mat'],'denoisedI_uint16_cut','denoisedI_doub_cut'); 
   
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

figure; 
for i=1:length(imginds)
    k = imginds(i); 
    subplot(6,6,i); imm = im_uint16(:,:,k); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
    imshow(imm, cmapr); 
end
figure; 
for i=1:length(imginds)
    k = imginds(i); 
    subplot(6,6,i); imm = denoisedI_uint16(:,:,k); cmapr = [quantile(imm(:),0.01) quantile(imm(:),0.99)];
    imshow(imm, cmapr); 
end

% figure; 
% subplot(1,3,1); [imtemp,map] = imread(nam, 20); imshow(imtemp,map);
% subplot(1,3,2); imshow(denoisedI,[min(denoisedI(:)) max(denoisedI(:))]); 
% subplot(1,3,3); imshow(denoisedI2,[min(denoisedI2(:)) max(denoisedI2(:))]); 

%% Run CaImAn

addpath(caimanpath); 
addpath(genpath([caimanpath,'utilities'])); 
addpath(genpath([caimanpath,'deconvolution'])); 

Y = cast(denoisedI_uint16_cut,'single'); 
 
%Y = Y - min(Y(:)); 
if ~isa(Y,'single');    Y = single(Y);  end         % convert to single
 
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% Set parameters
 
K = 25;                                           % number of components to be found
tau = 10;                                          % std of gaussian kernel (half size of neuron) 
p = 2; 
 
options = CNMFSetParms('d1',d1,'d2',d2,'p',p,'gSig',tau,'merge_thr',0.80,'nb',2,'min_SNR',3,'space_thresh',0.5,...
    'cnn_thr',0.2,'init_method','greedy');  
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
refine_components = false;  % flag for manual refinement
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

plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

%% make movie
if (0)  
    make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)
end





