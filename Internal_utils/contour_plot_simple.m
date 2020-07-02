function [CC] = contour_plot_simple(Aor,Cn,options,display_numbers,line_clr,text_clr,clrmap,ha)


% save and plot the contour traces of the found spatial components against
% a specified background image. The contour can be determined in two ways:
% options.thr_method = 'max': For every component pixel values below
%       options.thr_method are discarded
% options.thr_method = 'nrg': The contour is drawn around the value above
%        which a specified fraction of energy is explained (default 99%)

% INPUTS:
% Aor:              set of spatial components (matrix d x K)
% Cn:               background image (matrix d1 x d2)
% options:          options structure (optional)
% display_number:   flag for displaying the numbers of the components (optional, default: 0)
% line_clr:         Color of contour (default: red)
% text_clr:         Color of text numbering ROIs (default: black)
% clrmap:           Colormap of image (default: parula)
% ha:               Axes handle (if empty will plot in available graphics handle)

% OUTPUTS:
% CC:               contour plots coordinates

% Author: Suva Roy
% Date: 04/15/2020


if isempty(ha)
    ha = gca; 
end
axes(ha);
if ~exist('line_clr','var')
    line_clr = 'r'; 
end
if ~exist('text_clr','var')
    text_clr = 'k'; 
end
if ~exist('clrmap','var')
    cmap = parula(size(Aor,2)+5); %can be 'hot'
else 
    cmap = colormap(clrmap);
end
if ~exist('options','var') || (exist('options','var') && ~isfield(options,'maxthr'))
    options.maxthr = 0.25; % default value
end


imshow(Cn,[quantile(Cn(:),0.001) quantile(Cn(:),0.999)]); hold on; 
colormap(cmap); 
set(gca,'xtick',[],'ytick',[]); 


[d1,d2] = size(Cn);
ind_show = 1:size(Aor,2);
thr = options.maxthr;
for i = 1:size(Aor,2)
    A_temp = full(reshape(Aor(:,i),d1,d2));
    A_temp = medfilt2(A_temp,[3,3]);
    if sum(A_temp(:)==0)
        A_temp = full(reshape(Aor(:,i),d1,d2));     %-------------- modified by Suva Roy. 04/15/2020 --------------$=%
    end
    A_temp(A_temp<thr*max(A_temp(:))) = 0;
    BW = bwareafilt(A_temp>0,1);                
    BW2 = bwboundaries(BW);
    if ~isempty(BW2)
        for ii = 1:length(BW2)
            BW2{ii} = fliplr(BW2{ii});
            plot(BW2{ii}(:,1),BW2{ii}(:,2),'Color',line_clr, 'linewidth', 2);
        end
        CC{i} = BW2{1}';
        fp = find(BW);
        [ii,jj] = ind2sub([d1,d2],fp);
        CR{i,1} = [ii,jj]';
        CR{i,2} = A_temp(fp)';
    end
    hold on;
end

cm=com(Aor(:,1:end),d1,d2); % center of mass 
if display_numbers
    lbl = strtrim(cellstr(num2str((1:size(Aor,2))')));
    for i = ind_show
        text(round(cm(i,2)),round(cm(i,1)),strtrim(cellstr(num2str(i))),'color',text_clr,'fontsize',16,'fontweight','normal');
    end
end


