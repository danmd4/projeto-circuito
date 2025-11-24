import numpy as np

def fatoracao_lup(A_in):

    U = A_in.copy().astype(float)
    n = len(U)
    

    L = np.eye(n) 
    P = np.eye(n) 

    for k in range(n - 1): 
        
       
        p = np.argmax(np.abs(U[k:, k])) + k 
        
      
        if np.isclose(U[p, k], 0):
            raise ValueError("Erro: Matriz singular detectada.")
            
       
        U[[k, p]] = U[[p, k]]
        
       
        P[[k, p]] = P[[p, k]]
        
     
        L[[k, p], :k] = L[[p, k], :k]
        

        for i in range(k + 1, n): 
            L[i, k] = U[i, k] / U[k, k]
            
            
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            
    
    U = np.triu(U)
            
    return L, U, P 

def substituicao_direta(L, b_prime):
  
    n = len(b_prime)
    y = np.zeros(n)
    
    for i in range(n):
        soma = sum(L[i, j] * y[j] for j in range(i))
        y[i] = b_prime[i] - soma
       
        
    return y

def substituicao_retroativa(U, y):
  
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        soma = sum(U[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - soma) / U[i, i]
        
    return x

def resolver_sistema_lup(A, b):
   

    L, U, P = fatoracao_lup(A)
    
 
    b_prime = np.dot(P, b)
    

    y = substituicao_direta(L, b_prime)
    

    x = substituicao_retroativa(U, y)
    
    return x, L, U, P



if __name__ == "__main__":

    A = np.array([
        [1.0,     -1.0,    0.0,     0.0],
        [-0.3333,  0.8333, 0.0,    -0.3333],
        [0.0,     -0.1667, 0.4167, -0.1667],
        [0.0,      1.3333, 0.0,     1.0]
    ])


    b = np.array([20.0, 0.0, 0.0, 0.0])

    print("--- Resolvendo Circuito (Exemplo 9) ---")
    print(f"Matriz A:\n{A}")
    print(f"Vetor b: {b}\n")


    x_calc, L_calc, U_calc, P_calc = resolver_sistema_lup(A, b)

    print("--- Resultados da Fatoração ---")
    print(f"Matriz P (Permutação):\n{P_calc}\n")
    print(f"Matriz L (Inferior):\n{np.round(L_calc, 4)}\n")
    print(f"Matriz U (Superior):\n{np.round(U_calc, 4)}\n")

    print("--- Solução Final (Tensões Nodais) ---")

    print(f"Solução calculada x: {np.round(x_calc, 4)}")
    

    residuo = np.dot(A, x_calc) - b
    norma_residuo = np.linalg.norm(residuo)
    print(f"\nNorma do Resíduo ||Ax - b||: {norma_residuo:.4e}")