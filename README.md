# HK Matrix (GUI)

Aplicación de Álgebra Lineal con interfaz moderna (CustomTkinter).

Funciones:
- Operaciones con matrices: suma, resta y producto (A·B)
- Independencia de vectores (rango de columnas)
- Matriz Triangular Superior (triu)
- Forma Escalonada Reducida (RREF)
- Transpuesta e Inversa
- Determinante

## Requisitos
- Python 3.10+
- Paquetes: numpy, sympy, customtkinter

Ya han sido instalados en el entorno local del proyecto.

## ¿Cómo ejecutar?
En PowerShell, desde la carpeta `MATRIX`:

```powershell
C:/Users/hower/Desktop/MATRIX/.venv/Scripts/python.exe .\matrix.py
```

## Formatos de entrada
Puedes pegar matrices/vectores de varias formas:
- Estilo lista: `[[1,2,3],[4,5,6]]`
- Filas con espacios o comas: `1 2 3\n4 5 6` o `1,2,3\n4,5,6`
- Filas separadas con `;`: `1 2; 3 4`

Vectores (independencia): un vector por línea, por ejemplo:
```
1 0 0
0 1 0
0 0 1
```
O como lista: `[[1,0,0],[0,1,0],[0,0,1]]`

## Consejos
- Para RREF se usa SymPy; se muestran pivotes.
- La inversa solo existe si la matriz es cuadrada y tiene determinante distinto de 0.
- Cambia el tema (light/dark/system) desde el selector del panel izquierdo.

## Solución de problemas
- "No module named customtkinter":
  ```powershell
  C:/Users/hower/Desktop/MATRIX/.venv/Scripts/python.exe -m pip install customtkinter
  ```
- Si al ejecutar no aparece ventana, revisa si hay bloqueos del sistema o monitores virtuales.
