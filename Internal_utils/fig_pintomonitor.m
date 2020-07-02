function figh = fig_pintomonitor(figh, varargin)

    % Specify list of optional parameters
    p = inputParser;
    addParameter(p,'which_monitor', 'current', @ischar);
    addParameter(p,'outer_position', [], @isnumeric);
    addParameter(p,'units', 'normalized', @ischar);
    addParameter(p,'aspect_ratio_x_y', 4/3, @isnumeric);
    addParameter(p,'fracx', 0.4, @isnumeric); 
    p.parse(varargin{:});
    figprms = p.Results;

    % Search and assign figure handle 
    if ~exist('figh','var') || (exist('figh','var') && ~isa(figh, 'matlab.ui.Figure'))
        figh = figure;
    end

    % Get number of monitors 
    posgather = get(0,'MonitorPositions'); 
    num_monitors = size(posgather,1); 

    % Get positions of Matlab window 
    desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
    desktopMainFrame = desktop.getMainFrame;
    boundstruct = desktopMainFrame.getBounds; 
    bounds = abs([boundstruct.getX boundstruct.getY boundstruct.getWidth boundstruct.getHeight]);
    polymatlab = polyshape([bounds(1) bounds(1)+bounds(3) bounds(1)+bounds(3) bounds(1) bounds(1)],...
        [bounds(2) bounds(2) bounds(2)+bounds(4) bounds(2)+bounds(4) bounds(2)]);

    % Set relative size of figure (aspect ratio = 4:3)  
    fracx = figprms.fracx; 
    if isempty(figprms.aspect_ratio_x_y)
        figprms.aspect_ratio_x_y = 4/3; 
    end
    aspect_ratio_x_y = figprms.aspect_ratio_x_y; 
    fracy = fracx/aspect_ratio_x_y; 


    % Get monitor to pin figure in (max 2 monitors allowed currently) 
    switch num_monitors 
        case 1
            monitorcent = round([(posgather(1,1)+posgather(1,3))/2 (posgather(1,2)+posgather(1,4))/2]); 
            width = round(fracx*posgather(1,3)); 
            height = round(fracy*posgather(1,4));
            figouterpos = [monitorcent(1)-floor(width/2) monitorcent(2)-floor(height/2) ...
                width height]; 
        case 2
            polysectarea = zeros(2,1);
            for k=1:num_monitors
                polygx = [posgather(k,1) posgather(k,1)+posgather(k,3) posgather(k,1)+posgather(k,3) posgather(k,1) posgather(k,1)];
                polygy = [posgather(k,2) posgather(k,2) posgather(k,2)+posgather(k,4) posgather(k,2)+posgather(k,4) posgather(k,2)];
                polymonitor = polyshape(polygx, polygy);
                polysectarea(k) = area(intersect(polymonitor,polymatlab));
            end
            [~,indxMatlabMonitor] = max(polysectarea);
            if ~strcmpi(figprms.which_monitor,'Current') 
                indxMatlabMonitor = setxor(1:num_monitors,indxMatlabMonitor); 
            end
            monitorcent = round([posgather(indxMatlabMonitor,1)+posgather(indxMatlabMonitor,3)/2 ...
                posgather(indxMatlabMonitor,2)+posgather(indxMatlabMonitor,4)/2]); 
            width = round(fracx*posgather(indxMatlabMonitor,3)); 
            height = (1/aspect_ratio_x_y)*width; 
            
            figouterpos = [monitorcent(1)-floor(width/2) monitorcent(2)-floor(height/2) ...
                width height]; 
    end
    
    set(figh,'outerposition',figouterpos);
    figh;
end
    


