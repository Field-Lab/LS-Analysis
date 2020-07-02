function  [stim_xtrace, stim_ytrace] = make_calcium_movie(A, b, C, F_dff, vidpath, options, varargin)

% create movie of temporal trace, calcium response of ROIs and stimuli 

% INPUTS:
% A:        spatial footprints  (d x nr matrix)
% b:        spatial background  (d x 1 vector)
% C:        temporal components (nr X T matrix)
% F_dff:    detrended or dF/F0 trances (used for determining search locations only) 
% P:    dataset parameters (used for noise values and interpolated entries)

% options    parameter struct (for noise values and other parameters)

% OUTPUTS:
% A:    new estimate of spatial footprints
% b:    new estimate of spatial background
% C:    temporal components (updated only when spatial components are completely removed)

% Written by:
% Suva Roy, Duke University, 2020

numcomp = size(A,2);
num_images = size(C,2); 
d1 = options.d1; 
d2 = options.d2; 
fps = options.fr; 
backg = reshape(b,d1,d2);
t_axis = (1:num_images).*1/fps; % ms


sx = min([options.sx,floor(d1/2),floor(d2/2)]);
int_x = zeros(numcomp,2*sx);
int_y = zeros(numcomp,2*sx);
cm = com(A,d1,d2);
imfiles = cell(1,numcomp); 
for i=1:numcomp
    Atemp = reshape(A(:,i),d1,d2);
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

[stim_xtrace, stim_ytrace] = deal([]); 
if ~isempty(varargin{1}) 
    stim = varargin{1}; 
    if isstruct(stim)
        if strcmpi(stim.waveform,'pulse')
            numcycles = floor(t_axis(end)/stim.period); 
            stim_ytrace = repmat([0 1 1 0],1,numcycles).*stim.amplitude; 
            for nc=1:numcycles 
                stim_xtrace = [stim_xtrace [0 0 stim.pulse_width stim.pulse_width]+(nc-1)*stim.period]; 
            end
            stim_xtrace_1cycle = [0 0 stim.pulse_width stim.pulse_width 0]; 
            stim_ytrace_1cycle = [0 1 1 0 0].*stim.amplitude;
        elseif strcmpi(stim.waveform,'square')
            numcycles = floor(t_axis(end)/stim.period); 
            stim_ytrace = repmat([0 1 1 0],1,numcycles).*stim.amplitude; 
            for nc=1:numcycles 
                stim_xtrace = [stim_xtrace [0 0 stim.period/2 stim.period/2]+(nc-1)*stim.period]; 
            end            
            stim_xtrace_1cycle = [0 0 stim.period/2 stim.period/2 0]; 
            stim_ytrace_1cycle = [0 1 1 0 0].*stim.amplitude;
        elseif strcmpi(stim.waveform,'sine')
            numcycles = floor(t_axis(end)/stim.period);
            stim_ytrace = repmat(sin(2.*pi.*(0:0.01:stim.period-0.01)./stim.period).*stim.amplitude,1,numcycles);
            stim_xtrace = linspace(0,numcycles.*stim.period, length(stim_ytrace)); 
            stim_ytrace_1cycle = sin(2.*pi.*(0:0.01:stim.period)./stim.period).*stim.amplitude;
            stim_xtrace_1cycle = linspace(0,stim.period, length(stim_ytrace_1cycle)); 
        end
    end
end
t_axis_modStimPer = mod(t_axis, stim.period); 

if ~isempty(varargin{2})
    make_avi = varargin{2}; 
    if ~islogical(make_avi)
        make_avi = false; 
        warning('Skipping AVI creation!'); 
    end
end

if ~isempty(varargin{3})
    fig_aspect_ratio = varargin{3}; 
end



ln_cl = 'r';
txt_cl = 'y'; 
clrmap = flipud(cbrewer('seq', 'YlGnBu', 100,'pchip'));

if make_avi
    vidObj = VideoWriter(vidpath);
    set(vidObj,'FrameRate',15);
    open(vidObj);
end

ncols = 4; 
ncols1 = ceil(ncols/2); ncols2 = ncols1 + ceil(ncols1-ncols/2); 
ncols = ncols1 + ncols2; 
nrows = ncols1+1+numcomp; 

hf = fig_pintomonitor('','fracx',0.8, 'aspect_ratio_x_y',fig_aspect_ratio); clear ha; 

[locs1, locs2] = deal([]); 
for i=1:ncols1
    locs1 = [locs1 (i-1)*ncols+1:(i-1)*ncols+ncols1]; 
    locs2 = [locs2 (i-1)*ncols+1+ncols1:(i-1)*ncols+ncols]; 
end

% Top left: Active ROIs
ha(1) = subaxis((ncols1+numcomp+1),ncols, locs1, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
contour_plot_simple(A,backg,options,true,ln_cl,txt_cl,clrmap,ha(1)); 
title(ha(1), 'Active ROIs');

% Top right: Spatial background 
ha(2) = subaxis((ncols1+numcomp+1),ncols, locs2, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
montage(backg,'DisplayRange', [],'size',[1 1]); colormap(clrmap); 
title(ha(2), 'Spatial background');

% Below top right: Full stim trace 
ha(3) = subaxis((ncols1+numcomp+1),ncols, [ncols*ncols1+2:ncols*(ncols1+1)-1], 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
plot(ha(3), stim_xtrace, stim_ytrace, '-r','linewidth',1); box off; 
ylabel(ha(3),'Stim amplitude'); title(ha(3),sprintf('Full stim trace: %s',stim.waveform)); 

% Below top right: One period stim trace 
ha(4) = subaxis((ncols1+numcomp+1),ncols, ncols*(ncols1+1), 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
plot(ha(4), stim_xtrace_1cycle, stim_ytrace_1cycle, '-r','linewidth',1); box off; 
ylabel(ha(4),'Stim amplitude'); title(ha(4),'Stim: 1 cycle');   

% Bottom panels: ROI footprint (left), Full trace (mid-right), Repeat trace (extreme right)
for i=1:numcomp 
    locid = ncols*(ncols1+1)+1+(i-1)*ncols; 
    ha(locid) = subaxis((ncols1+numcomp+1),ncols, ncols*(ncols1+1) + 1 + (i-1)*ncols,'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
    montage(imfiles{i},'DisplayRange', [],'size',[1 1]); colormap(gray); 
end
parfor_progress(num_images-1); 
for j=1:num_images-1
    for i=1:numcomp
        locid = ncols*(ncols1+1) + 2 + (i-1)*ncols;
        locidend = ncols*(ncols1+1) + (ncols -1) + (i-1)*ncols;
        if j==1
            ha(locid) = subaxis((ncols1+numcomp+1),ncols, locid:locidend, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
            box(ha(locid),'off'); 
            hold(ha(locid), 'on'); 
        end
        y2 = F_dff(i,:);
        plot(ha(locid), t_axis(j:j+1), y2(j:j+1)./max(y2(:)), '-b','linewidth',1); 
        set(ha(locid),'xlim',[0 t_axis(end)],'ylim',[min(y2) max(y2)]); 
        title(ha(locid),'detrended dF/F0');
        
        
        locid = ncols*(ncols1+1) + (ncols) + (i-1)*ncols;
        if j==1
            ha(locid) = subaxis((ncols1+numcomp+1),ncols, locid, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);
            box(ha(locid),'off'); 
            hold(ha(locid), 'on'); 
        end
        if t_axis_modStimPer(j+1)<t_axis_modStimPer(j)
            plot(ha(locid), t_axis_modStimPer(j+1), y2(j+1)./max(y2(:)), '-b','linewidth',1); 
        else
            plot(ha(locid), t_axis_modStimPer(j:j+1), y2(j:j+1)./max(y2(:)), '-b','linewidth',1); 
        end
        set(ha(locid),'xlim',[0 stim.period],'ylim',[min(y2) max(y2)]);
        title(ha(locid),'Repeats: detrended dF/F0'); 
        
        drawnow; 
        
        if make_avi  
            currFrame = getframe(hf);
            writeVideo(vidObj,currFrame);    
        end
    end
    parfor_progress();
end
parfor_progress(0); 

if make_avi
    close(vidObj);
end

