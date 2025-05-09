try
    % Agregar el path donde está la función getBehavData
    addpath(genpath('C:\Users\Lak Lab\Documents\Github\sideBiasLateralisation\LakLabAnalysis'));
    
    % Verificar que la función existe
    if exist('getBehavData', 'file')
        disp('Función getBehavData encontrada');
    else
        disp('Función getBehavData NO encontrada');
        return;
    end
    
    % Intentar cargar el archivo Block.mat
    blockFile = 'Z:\OFZ011\2023-05-31\1\2023-05-31_1_OFZ011_Block.mat';
    if exist(blockFile, 'file')
        disp('Archivo Block.mat encontrado');
        % Intentar cargar el archivo
        blockData = load(blockFile);
        disp('Archivo Block.mat cargado correctamente');
        
        % Intentar llamar a getBehavData
        sessionName = '2023-05-31_1_OFZ011';
        sessionProfile = 'Value2AFC_noTimeline';
        try
            % Convertir los argumentos a cell arrays de MATLAB
            sessionName = {sessionName};
            sessionProfile = {sessionProfile};
            
            % Llamar a la función
            data = getBehavData(sessionName{1}, sessionProfile{1});
            disp('getBehavData ejecutado correctamente');
            
            % Mostrar información sobre los datos
            disp('Estructura de los datos:');
            disp(fieldnames(data));
        catch e
            disp('Error al ejecutar getBehavData:');
            disp(e.message);
            disp('Stack trace:');
            disp(e.stack);
        end
    else
        disp('Archivo Block.mat no encontrado');
    end
catch e
    disp('Error general:');
    disp(e.message);
    disp('Stack trace:');
    disp(e.stack);
end 
