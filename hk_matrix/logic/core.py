from __future__ import annotations
import numpy as np
from sympy import Matrix, Rational

def parse_matrix(text: str) -> np.ndarray:
    import ast, re
    s = text.strip()
    if not s:
        raise ValueError("Entrada vacía")
    try:
        candidate = ast.literal_eval(s.replace(";", ","))
        arr = np.array(candidate, dtype=float)
        if arr.ndim != 2:
            raise ValueError("La entrada debe representar una matriz 2D")
        return arr
    except Exception:
        pass
    rows = []
    for line in filter(None, [part.strip() for part in re.split(r"[;\n]+", s)]):
        tokens = [t for t in re.split(r"[\s,]+", line) if t]
        rows.append([float(t) for t in tokens])
    if not rows:
        raise ValueError("No se pudo interpretar la matriz")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("Todas las filas deben tener el mismo número de columnas")
    return np.array(rows, dtype=float)

def parse_vectors(text: str) -> np.ndarray:
    import ast, re
    s = text.strip()
    if not s:
        raise ValueError("Entrada vacía")
    try:
        candidate = ast.literal_eval(s.replace(";", ","))
        arr = np.array(candidate, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError
        return arr.T
    except Exception:
        pass
    vecs = []
    for line in filter(None, [part.strip() for part in re.split(r"[;\n]+", s)]):
        tokens = [t for t in re.split(r"[\s,]+", line) if t]
        vecs.append([float(t) for t in tokens])
    if not vecs:
        raise ValueError("No se pudo interpretar los vectores")
    dim = len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError("Todos los vectores deben tener la misma dimensión")
    return np.array(vecs, dtype=float).T

def fmt_num(x: float, decimals: int = 2) -> str:
    """Formato compacto: redondea a `decimals` y omite ceros si es entero."""
    v = round(float(x), decimals)
    if abs(v - int(round(v))) < 10**(-decimals):
        return str(int(round(v)))
    s = f"{v:.{decimals}f}".rstrip('0').rstrip('.')
    return s

def fmt_matrix(arr: np.ndarray, precision: int = 2) -> str:
    """Formatea matriz con números redondeados y sin ceros innecesarios."""
    if arr.size == 0:
        return "[]"
    rows = []
    for i in range(arr.shape[0]):
        rows.append("[" + ", ".join(fmt_num(arr[i, j], precision) for j in range(arr.shape[1])) + "]")
    return "[" + ("\n ".join(rows)) + "]"

def rref_steps(A: np.ndarray):
    M = Matrix([[Rational(x) for x in row] for row in A.tolist()])
    steps = [("Matriz inicial:", M.copy())]
    rows, cols = M.rows, M.cols
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = None
        for i in range(r, rows):
            if M[i, c] != 0:
                piv = i; break
        if piv is None:
            continue
        if piv != r:
            M.row_swap(piv, r)
            steps.append((f"Intercambiar fila {piv+1} con fila {r+1}", M.copy()))
        if M[r, c] != 1:
            factor = M[r, c]
            M.row_op(r, lambda v, j: v / factor)
            steps.append((f"Dividir fila {r+1} por {factor}", M.copy()))
        for i in range(rows):
            if i != r and M[i, c] != 0:
                factor = M[i, c]
                M.row_op(i, lambda v, j: v - factor * M[r, j])
                steps.append((f"R{i+1} <- R{i+1} - ({factor})*R{r+1}", M.copy()))
        r += 1
    steps.append(("Resultado: RREF", M.copy()))
    return steps

def add_steps(A: np.ndarray, B: np.ndarray):
    r, c = A.shape
    C = np.zeros((r, c), dtype=float)
    steps = [("Matriz resultado inicial (ceros)", Matrix(C.tolist()))]
    for i in range(r):
        for j in range(c):
            a = float(A[i, j]); b = float(B[i, j])
            val = a + b
            C[i, j] = val
            desc = (
                f"Calcular C[{i+1},{j+1}] = {fmt_num(a,2)} [{i+1},{j+1}] + "
                f"{fmt_num(b,2)} [{i+1},{j+1}] = {fmt_num(val,2)}"
            )
            steps.append((desc, Matrix(C.tolist())))
    steps.append(("Suma completa A + B", Matrix(C.tolist())))
    return steps

def sub_steps(A: np.ndarray, B: np.ndarray):
    r, c = A.shape
    C = np.zeros((r, c), dtype=float)
    steps = [("Matriz resultado inicial (ceros)", Matrix(C.tolist()))]
    for i in range(r):
        for j in range(c):
            a = float(A[i, j]); b = float(B[i, j])
            val = a - b
            C[i, j] = val
            desc = (
                f"Calcular C[{i+1},{j+1}] = {fmt_num(a,2)} [{i+1},{j+1}] - "
                f"{fmt_num(b,2)} [{i+1},{j+1}] = {fmt_num(val,2)}"
            )
            steps.append((desc, Matrix(C.tolist())))
    steps.append(("Resta completa A - B", Matrix(C.tolist())))
    return steps

def multiply_steps(A: np.ndarray, B: np.ndarray):
    r, n = A.shape
    _, c = B.shape
    C = np.zeros((r, c), dtype=float)
    steps = [("Matriz resultado inicial (ceros)", Matrix(C.tolist()))]
    for i in range(r):
        for j in range(c):
            val = float(np.sum([A[i,k]*B[k,j] for k in range(n)]))
            C[i,j] = val
            terms = [f"{A[i,k]:.2f}*{B[k,j]:.2f}" for k in range(n)]
            desc = f"Calcular C[{i+1},{j+1}] = " + " + ".join(terms) + f" = {val:.2f}"
            steps.append((desc, Matrix(C.tolist())))
    steps.append(("Producto completo A·B", Matrix(C.tolist())))
    return steps

def upper_triangular_steps(A: np.ndarray):
    M = Matrix([[Rational(x) for x in row] for row in A.tolist()])
    steps = [("Matriz inicial", M.copy())]
    rows, cols = M.rows, M.cols
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = None
        for i in range(r, rows):
            if M[i, c] != 0:
                piv = i; break
        if piv is None:
            continue
        if piv != r:
            M.row_swap(piv, r)
            steps.append((f"Intercambiar fila {piv+1} con fila {r+1}", M.copy()))
        for i in range(r+1, rows):
            if M[i, c] != 0:
                factor = M[i, c] / M[r, c]
                M.row_op(i, lambda v, j: v - factor * M[r, j])
                steps.append((f"R{i+1} <- R{i+1} - ({factor})*R{r+1}", M.copy()))
        r += 1
    steps.append(("Resultado: U (triangular superior)", M.copy()))
    return steps

def transpose_steps(A: np.ndarray):
    M = Matrix(A.tolist())
    steps = [("Matriz original A", M.copy())]
    steps.append(("Transponer: A^T (filas↔columnas)", M.T.copy()))
    return steps

def inverse_steps(A: np.ndarray):
    # Método por matriz aumentada [A|I] y operaciones elementales
    if A.shape[0] != A.shape[1]:
        return [("La matriz no es cuadrada, no existe inversa.", Matrix(A.tolist()))]
    M = Matrix([[Rational(x) for x in row] for row in A.tolist()])
    n = M.rows
    Aug = M.row_join(Matrix.eye(n))
    steps = [("Matriz aumentada [A|I]", Aug.copy())]
    r = 0
    for c in range(n):
        if r >= n: break
        piv = None
        for i in range(r, n):
            if Aug[i, c] != 0:
                piv = i; break
        if piv is None: continue
        if piv != r:
            Aug.row_swap(piv, r)
            steps.append((f"Intercambiar fila {piv+1} con fila {r+1}", Aug.copy()))
        if Aug[r, c] != 1:
            factor = Aug[r, c]
            Aug.row_op(r, lambda v, j: v / factor)
            steps.append((f"Dividir fila {r+1} por {factor}", Aug.copy()))
        for i in range(n):
            if i != r and Aug[i, c] != 0:
                factor = Aug[i, c]
                Aug.row_op(i, lambda v, j: v - factor * Aug[r, j])
                steps.append((f"R{i+1} <- R{i+1} - ({factor})*R{r+1}", Aug.copy()))
        r += 1
    left = Aug[:, :n]
    right = Aug[:, n:]
    if left == Matrix.eye(n):
        steps.append(("Izquierda = I: la derecha es A^{-1}", Aug.copy()))
    else:
        steps.append(("La izquierda no es I ⇒ A no es invertible", Aug.copy()))
    return steps

def cramer_steps(A: np.ndarray, b: np.ndarray):
    """Resuelve Ax = b utilizando el método de Cramer.

    Devuelve una tupla (solucion, pasos, detA, det_columnas, solucion_exacta)
    donde "solucion" es un arreglo numpy con la aproximación decimal del vector
    x, "pasos" es la bitácora simbólica para mostrar en la UI, "detA" es el
    determinante de la matriz de coeficientes, "det_columnas" contiene los
    determinantes de las matrices A_i y "solucion_exacta" la solución simbólica.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.ndim != 2 or b.shape[1] != 1:
        raise ValueError("El vector b debe tener una sola columna.")
    n, m = A.shape
    if n != m:
        raise ValueError("La matriz de coeficientes debe ser cuadrada.")
    if b.shape[0] != n:
        raise ValueError("El vector b debe tener tantas filas como A.")

    def _to_rational_matrix(mat: np.ndarray) -> Matrix:
        return Matrix([[Rational(str(val)) for val in row] for row in mat.tolist()])

    coeff = _to_rational_matrix(A)
    vec = Matrix([[Rational(str(val))] for val in b.flatten().tolist()])
    augmented = coeff.row_join(vec)
    steps = [("Sistema aumentado [A|b]", augmented.copy())]

    detA = coeff.det()
    steps.append((f"det(A) = {detA}", coeff.copy()))
    if detA == 0:
        steps.append(("det(A) = 0 ⇒ el método de Cramer no aplica (no hay solución única)", coeff.copy()))
        return None, steps, detA, [], []

    det_columnas = []
    solucion_exacta = []
    for idx in range(n):
        Ai = coeff.copy()
        Ai[:, idx] = vec
        steps.append((f"A_{idx+1}: reemplazar columna {idx+1} por b", Ai.copy()))
        detAi = Ai.det()
        det_columnas.append(detAi)
        steps.append((f"det(A_{idx+1}) = {detAi}", Ai.copy()))
        sol_i = detAi / detA
        solucion_exacta.append(sol_i)
        steps.append((f"x_{idx+1} = det(A_{idx+1}) / det(A) = {detAi}/{detA}", Matrix([[sol_i]])))

    vector_sol = Matrix(solucion_exacta)
    steps.append(("Vector solución x", vector_sol.copy()))
    solucion = np.array([[float(val.evalf())] for val in solucion_exacta], dtype=float)
    return solucion, steps, detA, det_columnas, solucion_exacta


def determinant_steps(A: np.ndarray):
    # Eliminación hacia triangular superior sin escalar filas
    M = Matrix([[Rational(x) for x in row] for row in A.tolist()])
    steps = [("Matriz inicial", M.copy())]
    rows, cols = M.rows, M.cols
    if rows != cols:
        steps.append(("No es cuadrada ⇒ determinante no definido", M.copy()))
        return steps
    swaps = 0
    r = 0
    for c in range(cols):
        if r >= rows: break
        piv = None
        for i in range(r, rows):
            if M[i, c] != 0:
                piv = i; break
        if piv is None: continue
        if piv != r:
            M.row_swap(piv, r); swaps += 1
            steps.append((f"Swap filas {piv+1}↔{r+1} (cambia signo del det)", M.copy()))
        for i in range(r+1, rows):
            if M[i, c] != 0:
                factor = M[i, c] / M[r, c]
                M.row_op(i, lambda v, j: v - factor * M[r, j])
                steps.append((f"Eliminar debajo del pivote: R{i+1} <- R{i+1} - ({factor})*R{r+1}", M.copy()))
        r += 1
    det = Rational(1)
    for i in range(rows):
        det *= M[i, i]
    if swaps % 2 == 1:
        det = -det
    steps.append((f"Determinante = producto diagonal * (-1)^swaps = {det}", M.copy()))
    return steps
__all__ = [
    'parse_matrix','parse_vectors','fmt_matrix','fmt_num',
    'rref_steps','add_steps','sub_steps','multiply_steps','upper_triangular_steps',
    'transpose_steps','inverse_steps','determinant_steps','cramer_steps'
]
