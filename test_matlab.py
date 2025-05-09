import matlab.engine
print("Importando matlab.engine...")

try:
    eng = matlab.engine.start_matlab()
    print("Motor de MATLAB iniciado correctamente")
    
    # Probar una operación simple
    result = eng.sqrt(4.0)
    print(f"Resultado de sqrt(4.0): {result}")
    
except Exception as e:
    print("Error al iniciar MATLAB:")
    print(str(e)) 