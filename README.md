# Mínimos Quadrados Não Linear

Neste trabalho faremos uma análise do método dos Mínimos Quadrados aplicado a dois conjuntos de dados que sabemos representar uma função não linear. Quando linearizamos uma função não linear e aplicamos o método, estamos encontrando uma solução para uma combinação linear da função objeto de estudo, o que não indica uma solução ótima.

Consideremos que o conjunto de dados representa a distribuição Normal com média <em>a</em> e variância $\sigma^{2}$.

Os valores dos parâmetros que melhor aproximam o conjunto de dados (t,y) configura um problema de minimização.

Neste estudo aplicaremos tanto o método dos Mínimos Quadrados ao problema linearizado quanto o Método de Gauss-Newton para problemas de mínimos quadrados não lineares.

O método de Gauss-Newton pode ser visto como uma modificação do Método de Newton para encontrar o mínimo de uma função. Diferentemente do Método de Newton, ele apenas pode ser usado para minimizar uma soma dos valores quadrados da função, mas tem a vantagem de que as derivadas segundas, que podem ser difíceis de calcular, não são necessárias. 
