import numpy as np

class DoolittleSolver:
    """Implementação do método de Doolittle para decomposição LU"""
    
    def __init__(self, tol=1e-12):
        self.tol = tol
        self.L = None
        self.U = None
    
    def decompose(self, A):
        """Decomposição LU pelo método de Doolittle"""
        A = np.array(A, dtype=float)
        n = A.shape[0]
        L = np.eye(n)
        U = np.zeros((n, n))
        
        for j in range(n):
            # Calcular coluna j de U
            for i in range(j + 1):
                U[i,j] = A[i,j] - np.sum(L[i,:i] * U[:i,j])
            
            if abs(U[j,j]) < self.tol:
                raise ValueError("Matriz singular")
            
            # Calcular coluna j de L
            for i in range(j + 1, n):
                L[i,j] = (A[i,j] - np.sum(L[i,:j] * U[:j,j])) / U[j,j]
        
        self.L, self.U = L, U
        return L, U
    
    def solve(self, A, b):
        """Resolve AX = b usando decomposição LU"""
        b = np.array(b, dtype=float)
        
        if self.L is None:
            self.decompose(A)
        
        # Substituição progressiva: LY = b
        n = len(b)
        Y = np.zeros(n)
        for i in range(n):
            Y[i] = b[i] - np.sum(self.L[i,:i] * Y[:i])
        
        # Substituição regressiva: UX = Y
        X = np.zeros(n)
        for i in range(n-1, -1, -1):
            X[i] = (Y[i] - np.sum(self.U[i,i+1:] * X[i+1:])) / self.U[i,i]
        
        return X

def analisar_circuito():
    """Análise do circuito elétrico com fontes dependentes"""
    # Matriz do sistema
    A = np.array([
        [6, 0, 0, 0, -1, -1, 0],
        [0, 12, -8, 0, 0, 1, 0],
        [0, -8, 10, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0],
        [1, 0, 0, -1, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, -3],
        [0, 0, 1, 0, 0, 0, 1]
    ])
    
    # Vetor independente
    B = np.array([0, 0, -10, 0, 5, 0, 0])
    
    # Resolver sistema
    solver = DoolittleSolver()
    X = solver.solve(A, B)
    
    # Resultados
    variaveis = ['i1', 'i2', 'i3', 'i4', 'v5Λ', 'v3ix', 'ix']
    referencia = [-2.5000, 3.9300, 2.1400, -7.5000, 15.0000, -30.0000, -2.1400]
    
    print("Solução do Circuito:")
    for i, (var, x_val, ref) in enumerate(zip(variaveis, X, referencia)):
        erro = abs((x_val - ref) / ref * 100) if ref != 0 else 0
        print(f"{var}: {x_val:.4f} (ref: {ref:.4f}, erro: {erro:.2f}%)")
    
    # Verificação
    residuo = np.linalg.norm(A @ X - B)
    print(f"\nResíduo: {residuo:.2e}")

if __name__ == "__main__":
    analisar_circuito()
