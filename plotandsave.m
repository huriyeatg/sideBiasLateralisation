function plotandsave(session)
    % Visualize the training for the given session
    visualiseTrainingGrating2AFC(session, 'Grating2AFC_noTimeline', []);
    
    % Create the filename
    figname = [session, '.png'];

    % Define the path where the figure will be saved
    filePath = fullfile('//QNAP-AL001.dpag.ox.ac.uk/CMedina/cingulate_DMS/Instrumental_TR/', figname);
    
    % Save the figure to the specified path
    saveas(gcf, filePath);
    
    % Display the path where the figure is saved
    disp(['Saved to: ', filePath]);
end
