#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[87]:


"""
SME0892 - Cálculo Numérico para Estatística
Trabalho #2 - Mínimos Quadrados Não Linear

-Nome: Carolina Spera Braga - Número USP: 7161740
"""


# Bibliotecas utilizadas
import numpy as np
import scipy as sp
import matplotlib.pyplot as pt
import scipy.linalg as la
from scipy import linalg
from scipy.interpolate import lagrange


# Método de Gauss-Newton puro aplicado ao conjunto de dados 1  ###############################

"""Gostaríamos de ajustar o modelo da distribuição Normal, com função Gaussiana dada por:
y = (2*(np.pi)*s**2)**(-1/2))*np.exp((-(t-a)**2)/(2*(s**2))) 
aos seguintes dados, usando o método de Gauss-Newton."""

# Leitura do arquivo txt.
arq = open("gauss_data_51.txt", "r+")
    
# Organização dos pontos (t,y).
l = list(arq)
l1 = [i.replace("\n","").strip().split() for i in l]

t = np.array([float(i[0]) for i in l1])
y = np.array([float(i[1]) for i in l1])

# Definimos a função resíduo (como uma função de x=(a,s))
def residuo(x):
    return y - ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))

# Definimos a matriz Jacobiana da função resíduo
def jacobiano(x):
    return np.array([
        -((t-x[0])/((x[1]**3)*(2*np.pi)**(1/2)))*np.exp(-((t-x[0])**2)/(2*x[1]**2)),
        (np.exp(-((t-x[0])**2)/(2*x[1]**2))*(1/2+1/(x[1]**2)))/((x[1]**3)*(2*np.pi)**(1/2))
        ]).T

# Definimos uma função para plotar um gráfico que mostre o ajuste da função
# Gaussiana ao conjunto de dados e imprimir a norma da função resíduo para
# termos uma ideia se nosso chute inicial faz sentido
def grafico_residuo(x):
    pt.plot(t, y, 'ro', markersize=1, clip_on=False)
    T = np.linspace(t.min(), t.max(), 2000)
    Y = ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))
    pt.plot(T, Y, 'b-')
    
    return("Função r(x):", la.norm(residuo(x), 2)/2)


"""
Usamos numpy.linalg.lstsq() para resolver o problema dos mínimos quadrados, 
notando que a função retorna uma tupla na qual o primeiro termo é o que procuramos.
Imprimimos também a norma do resíduo.
"""

# Chute inicial
x = np.array([5, 1])
# Contagem do número de iterações necessárias com erro relativo < 10^(-6)
iteracoes = 0
# Definimos o número máximo de iterações em 1000 ou até satisfazer a condição do erro relativo
for k in range(1000):
#     pt.figure()
    x_inicio = x.copy()
    x = x + la.lstsq(jacobiano(x), -residuo(x))[0]
    grafico_residuo(x)
    iteracoes += 1
    pt.title("Iteração "+str(iteracoes)+ " com chute inicial (5,1)")
#     print(iteracoes, "a: ",x[0],"e sigma:", x[1])
#     print(grafico_residuo(x))
    # Imposição do limite da tolerância
    if np.linalg.norm(x - x_inicio, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < 10**(-6):
        break

# Impressão dos resultados
print("Número de iterações: ", iteracoes)
print("Coeficiente a: ", x[0])
print("Coeficiente sigma:", x[1])
print(grafico_residuo(x))


# In[81]:


# Método de Gauss-Newton puro aplicado ao conjunto de dados 2  ###############################

"""Gostaríamos de ajustar o modelo da distribuição Normal, com função Gaussiana dada por:
y = (2*(np.pi)*s**2)**(-1/2))*np.exp((-(t-a)**2)/(2*(s**2))) 
aos seguintes dados, usando o método de Gauss-Newton."""

# Leitura do arquivo txt.
arq = open("gauss_data_22.txt", "r+")
    
# Organização dos pontos (t,y).
l = list(arq)
l1 = [i.replace("\n","").strip().split() for i in l]

t = np.array([float(i[0]) for i in l1])
y = np.array([float(i[1]) for i in l1])

# Definimos a função resíduo (como uma função de x=(a,s))
def residuo(x):
    return y - ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))

# Definimos a matriz Jacobiana da função resíduo
def jacobiano(x):
    return np.array([
        -((t-x[0])/((x[1]**3)*(2*np.pi)**(1/2)))*np.exp(-((t-x[0])**2)/(2*x[1]**2)),
        (np.exp(-((t-x[0])**2)/(2*x[1]**2))*(1/2+1/(x[1]**2)))/((x[1]**3)*(2*np.pi)**(1/2))
        ]).T

# Definimos uma função para plotar um gráfico que mostre o ajuste da função
# Gaussiana ao conjunto de dados e imprimir a norma da função resíduo para
# termos uma ideia se nosso chute inicial faz sentido
def grafico_residuo(x):
    pt.plot(t, y, 'ro', markersize=1, clip_on=False)
    T = np.linspace(t.min(), t.max(), 3000)
    Y = ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))
    pt.plot(T, Y, 'b-')
    
    return("Função r(x):", la.norm(residuo(x), 2)/2)


"""
Usamos numpy.linalg.lstsq() para resolver o problema dos mínimos quadrados, 
notando que a função retorna uma tupla na qual o primeiro termo é o que procuramos.
Imprimimos também a norma do resíduo.
"""

# Chute inicial
x = np.array([2, 2])
# Contagem do número de iterações necessárias com erro relativo < 10^(-6)
iteracoes = 0
# Definimos o número máximo de iterações em 1000 ou até satisfazer a condição do erro relativo
for k in range(1000):
#     pt.figure()
    x_inicio = x.copy()
    x = x + la.lstsq(jacobiano(x), -residuo(x))[0]
    grafico_residuo(x)
    iteracoes += 1
    pt.title("Iteração "+str(iteracoes)+ " com chute inicial (5,1)")
#     print(iteracoes, "a: ",x[0],"e sigma:", x[1])
#     print(grafico_residuo(x))
    # Imposição do limite da tolerância
    if np.linalg.norm(x - x_inicio, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < 10**(-6):
        break

# Impressão dos resultados
print("Número de iterações: ", iteracoes)
print("Coeficiente a: ", x[0])
print("Coeficiente sigma:", x[1])
print(grafico_residuo(x))


# In[83]:


# Linearizando a Gaussinana para o conjunto de dados 1  #######################################

# Leitura do arquivo txt.
arq = open("gauss_data_51.txt", "r+")
    
# Organização dos pontos (t,y).
l = list(arq)
l1 = [i.replace("\n","").strip().split() for i in l]

t = np.array([float(i[0]) for i in l1])
y = np.array([float(i[1]) for i in l1])

# Matriz de Vandermonde
X = np.array([np.ones(2000),t,t**2,t**3,t**4]).transpose()  

# Encontramos os coeficientes a0, a1, a2, a3 e a4 com a função numpy.linalg.lstsq da scipy
coef = np.linalg.lstsq(X,np.log(y**2),rcond=None)[0]  

# Calculamos 'a' e 'sigma' a partir dos coeficientes calculados acima
sigma = (1/(4*coef[4]))**(1/2)
a = -coef[3]*sigma**2
print("a: ", a)
print("Sigma: ", sigma)


# In[84]:


# Linearizando a Gaussinana para o conjunto de dados 2  ########################################

# Leitura do arquivo txt.
arq2 = open("gauss_data_22.txt", "r+")
    
# Organização dos pontos (t,y).
l2 = list(arq2)
l12 = [i.replace("\n","").strip().split() for i in l2]

t2 = np.array([float(i[0]) for i in l12])
y2 = np.array([float(i[1]) for i in l12])

# Matriz de Vandermonde
X2 = np.array([np.ones(3000),t2,t2**2,t2**3,t2**4]).transpose()  

# Encontramos os coeficientes a0, a1, a2, a3 e a4 com a função numpy.linalg.lstsq da scipy
coef2 = np.linalg.lstsq(X2,np.log(y2**2),rcond=None)[0]  

# Calculamos 'a' e 'sigma' a partir dos coeficientes calculados acima
sigma = (1/(4*coef2[4]))**(1/2)
a = -coef2[3]*sigma**2
print("a: ", a)
print("Sigma: ", sigma)


# In[140]:


# Interpolação quadrática pela forma de Lagrange  ##############################################

# Pontos extremos e ponto médio de cada parâmetro
t = np.array([0.0,5.0025013,10.0])
y = np.array([-0.29081064,0.12505737,0.98522999])

# Obtenção dos coeficientes a0, a1 e a2 pela forma de Lagrande 
p = lagrange(t, y)
print(p) # P_2(t) = a0*t² + a1*t + a2


# In[145]:


# Método de Gauss-Newton amortecido aplicado ao conjunto de dados 1  ###########################

"""Gostaríamos de ajustar o modelo da distribuição Normal, com função Gaussiana dada por:
y = (2*(np.pi)*s**2)**(-1/2))*np.exp((-(t-a)**2)/(2*(s**2))) 
aos seguintes dados, usando o método de Gauss-Newton."""

# Leitura do arquivo txt.
arq = open("gauss_data_51.txt", "r+")
    
# Organização dos pontos (t,y).
l = list(arq)
l1 = [i.replace("\n","").strip().split() for i in l]

t = np.array([float(i[0]) for i in l1])
y = np.array([float(i[1]) for i in l1])

# Definimos a função resíduo (como uma função de x=(a,s))
def residuo(x):
    return y - ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))

# Definimos a matriz Jacobiana da função resíduo
def jacobiano(x):
    return np.array([
        -((t-x[0])/((x[1]**3)*(2*np.pi)**(1/2)))*np.exp(-((t-x[0])**2)/(2*x[1]**2)),
        (np.exp(-((t-x[0])**2)/(2*x[1]**2))*(1/2+1/(x[1]**2)))/((x[1]**3)*(2*np.pi)**(1/2))
        ]).T

# Definimos uma função para plotar um gráfico que mostre o ajuste da função
# Gaussiana ao conjunto de dados e imprimir a norma da função resíduo para
# termos uma ideia se nosso chute inicial faz sentido
def grafico_residuo(x):
    pt.plot(t, y, 'ro', markersize=1, clip_on=False)
    T = np.linspace(t.min(), t.max(), 2000)
    Y = ((2*(np.pi)*x[1]**2)**(-1/2))*np.exp((-(t-x[0])**2)/(2*(x[1]**2)))
    pt.plot(T, Y, 'b-')
    
    return("Função r(x):", la.norm(residuo(x), 2)/2)


"""
Usamos numpy.linalg.lstsq() para resolver o problema dos mínimos quadrados, 
notando que a função retorna uma tupla na qual o primeiro termo é o que procuramos.
Imprimimos também a norma do resíduo.
"""

# Chute inicial
x = np.array([5, 1])
# Contagem do número de iterações necessárias com erro relativo < 10^(-6)
iteracoes = 0
# Definimos o número máximo de iterações em 1000 ou até satisfazer a condição do erro relativo
for k in range(3000):
#     pt.figure()
    x_inicio = x.copy()
    x = x + la.lstsq(jacobiano(x), -residuo(x))[0] * 0.00889886
    grafico_residuo(x)
    iteracoes += 1
    pt.title("Iteração "+str(iteracoes)+ " com chute inicial (5,1)")
#     print(iteracoes, "a: ",x[0],"e sigma:", x[1])
#     print(grafico_residuo(x))
    # Imposição do limite da tolerância
    if np.linalg.norm(x - x_inicio, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < 10**(-6):
        break

# Impressão dos resultados
print("Número de iterações: ", iteracoes)
print("Coeficiente a: ", x[0])
print("Coeficiente sigma:", x[1])
print(grafico_residuo(x))

