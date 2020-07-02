%addpath(genpath('/Users/Suva/Desktop/matlab/'));
javaaddpath('/Applications/Vision.app/Contents/Resources/Java/Vision.jar');
if exist('/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Imaging/Light_sheet/Analysis/','dir')
    addpath(genpath('/Volumes/dusom_fieldlab/All_Staff/lab/Experiments/Imaging/Light_sheet/Analysis/'));
end
set(0,'DefaultFigureWindowStyle','normal')

% Link necessary files and scripts
caimanpath = '/Users/Suva/Documents/Projects/LightSheet/MatlabCodes/CaImAn-MATLAB-master/'; 
addpath(genpath(caimanpath)); 
addpath(genpath('/Users/Suva/Documents/Projects/LightSheet/MatlabCodes/CaImAn_analysis_codebase/')); 
addpath('/Users/Suva/Documents/MATLAB/');
%set(0,'DefaultFigureWindowStyle','docked'); 


