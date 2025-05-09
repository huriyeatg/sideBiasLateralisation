import matlab.engine
import os

# Iniciar el motor de MATLAB
eng = matlab.engine.start_matlab()

# Agregar el path de MATLAB usando la ruta absoluta
base_path = os.path.abspath('C:/Users/Lak Lab/Documents/Github/sideBiasLateralisation/LakLabAnalysis')
utility_path = os.path.join(base_path, 'Utility')
rigbox_path = os.path.abspath('C:/Users/Lak Lab/Documents/Github/sideBiasLateralisation/Rigbox')
burgbox_path = os.path.join(rigbox_path, 'cb-tools', 'burgbox')

# Agregar todos los directorios necesarios al path
eng.addpath(base_path)
eng.addpath(utility_path)
eng.addpath(rigbox_path)
eng.addpath(burgbox_path)

# Verificar que la función existe
try:
    eng.eval("which getBehavData", nargout=0)
    print("Función getBehavData encontrada")
except:
    print("ERROR: Función getBehavData no encontrada")
    print("Buscando en directorios:")
    print("Base:", base_path)
    print("Utility:", utility_path)
    print("Rigbox:", rigbox_path)
    print("Burgbox:", burgbox_path)
    eng.eval("path", nargout=0)
    exit(1)

# Definir los parámetros
sessionName = '2023-05-31_1_OFZ011'
sessionProfile = 'Value2AFC_noTimeline'

try:
    # Llamar a la función directamente con strings
    data = eng.getBehavData(sessionName, sessionProfile)
    print('getBehavData ejecutado correctamente')
    
except Exception as e:
    print('Error al ejecutar getBehavData:')
    print(str(e)) 