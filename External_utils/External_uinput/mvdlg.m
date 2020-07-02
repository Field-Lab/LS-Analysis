function Answer = mvdlg(prompt,title,position)
%MVDLG moveable input dialog box.
%  ANSWER = MVDLG(PROMPT,TITLE,POSITION) creates a dialog box that returns
%  user input for a prompt in cell array ANSWER. PROMPT is a string. TITLE
%  is a string that species a title for the dialog box. POSITION is a
%  four-element vector that specifies the size and location on the screen
%  of the dialog window. Specify POSITION in normalized coordinates in the
%  form: [left, bottom, width, height] where left and bottom are the
%  distance from the lower-left corner of the screen to the lower-left
%  corner of the dialog box. width and height are the dimensions.  
%
%
%  Examples:
%
%  prompt='Enter a value here:';
%  name='Data Entry Dialog';
%  position = [.5 .5 .2 .1];
%
%  answer=mvdlg(prompt,name,position);
%
%
%  prompt='Enter a value here:';
%  name='Data Entry Dialog';
%  position = [.7 .2 .2 .3]; 
%  answer=mvdlg(prompt,name,position);

%set up main window figure
dialogWind = figure('Units','normalized','Position', ...
    position,'MenuBar','none','NumberTitle','off','Name',title);

%set up GUI controls and text
uicontrol('style','text','String',prompt,'Units','normalized', ...
    'Position',[.05,.75,.9,.1],'HorizontalAlignment','left')
hedit = uicontrol('style','edit','Units','normalized','Position', ...
    [.05,.5,.9,.2],'HorizontalAlignment','left');
okayButton = uicontrol('style','pushbutton','Units','normalized',...
    'position', [.05,.1,.4,.3],'string','OK','callback',@okCallback);
cancelButton = uicontrol('style','pushbutton','Units','normalized',...
    'position', [.55,.1,.4,.3],'string','Cancel','callback',@cancCallback);

%initialize ANSWER to empty cell array
Answer = {};

%wait for user input, and close once a button is pressed 
uiwait(dialogWind);

%callbacks for 'OK' and 'Cancel' buttons
    function okCallback(hObject,eventdata)
        Answer = cell(get(hedit,{'String'}));
        uiresume(dialogWind);
        close(dialogWind);
    end

    function cancCallback(hObject,eventdata)
        uiresume(dialogWind);
        close(dialogWind);
    end
end
