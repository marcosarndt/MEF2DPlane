# Método dos Elementos Finitos para Problemas de Estado plano de tensão e deformação

import numpy as np
from scipy import linalg
# import marcha_tempolin as mtemp
import plotmef2d as pltmef
import ler_dados

# Matriz Constitutiva
def constitutiva(tpproblema,E,ni):
    if tpproblema == 'EPT':
        De = E/(1-ni**2) * np.array([[1,ni,0],[ni,1,0],[0,0,(1-ni)/2]])
    else:
        De = E/((1+ni)*(1-2*ni)) * np.array([[1-ni,ni,0],[ni,1-ni,0],[0,0,(1-2*ni)/2]])
    return(De)

# Entrada de dados
# Tipo de entrada de dados
string = input('Qual o caminho e o nome do arquivo de modelo que você deseja abrir (nome do arquivo sem extensão? ')
# Transforma a string de entrada em uma raw string
arquivo = string.replace("\\", "\\\\")

# Leitura de dados da malha Gmsh
Coord,Inci,tpel,list_mat,dict_cc = ler_dados.read_mesh2D(arquivo)
nnos = Coord.shape[0] # número de nós
nelem = Inci.shape[0] # número de elementos
nglt = 2*nnos # número total de graus de liberdade

# Plotando a malha 
pltmef.plot_malha2D(Coord,Inci)

# Leitura de dados de materiais e condições de contorno do arquivo .mcc
# Constrói o vetor de deslocamentos contendo os deslocamentos prescritos
# e o vetor de forças com forças nodais de superfície
dados_analise,Mat,list_cc,d,F = ler_dados.read_mccEP(arquivo,list_mat,dict_cc,nglt,Coord)
ncon = len(list_cc) # número de condições de contorno
dim = nglt - ncon # dimensão do problema após consideração das condições de contorno
print(f'C. de contorno: {list_cc}')
print(f'Vetor de deslocamentos: {d}')
print(f'Vetor de forças: {F}')

# Condições de Contorno Essenciais
ordem = [i for i in range(nglt)]
for i in range(ncon):
    ordem.remove(list_cc[i])
    ordem.append(list_cc[i])
print(f'Ordem: {ordem}')

# Matrizes de rigidez elementares e global
list_B = []
list_pos = []
Kg = np.zeros((nglt,nglt))
if tpel == 'triangle':
    for i in range(nelem):
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        Ae = 1/2.*(Coord[n2,0]*Coord[n3,1]+Coord[n1,0]*Coord[n2,1]+Coord[n3,0]*Coord[n1,1]-Coord[n2,0]*Coord[n1,1]
                   -Coord[n1,0]*Coord[n3,1]-Coord[n3,0]*Coord[n2,1])
        if Ae < 0: 
            print(f'Elemento {i}: Ae = {Ae}')
            parada = input()
        b = np.array([Coord[n2,1]-Coord[n3,1],(-1)*Coord[n1,1]+Coord[n3,1],Coord[n1,1]-Coord[n2,1]])
        c = np.array([(-1)*Coord[n2,0]+Coord[n3,0],Coord[n1,0]-Coord[n3,0],(-1)*Coord[n1,0]+Coord[n2,0]])
        B = 1/(2*Ae)*np.array([[b[0],0.,b[1],0.,b[2],0.],[0.,c[0],0.,c[1],0.,c[2]],[c[0],b[0],c[1],b[1],c[2],b[2]]])
        list_B.append(B)
        if dados_analise[0] == 'EPT':
            t = Mat[Inci[i,0],0]
        else:
            t = 1.
        Ke = t*Ae*np.dot(np.dot(np.transpose(B),D),B)
        pos = [2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1]
        list_pos.append(pos)
        for j in range(6):
            for k in range(6):
                Kg[pos[j],pos[k]] += Ke[j,k]
    # print(list_B)
    # print(list_pos)
    print(f'Matriz de rigidez global: {Kg}')

# Reordenação da matriz de rigidez e vetor de forças
Kgord = np.zeros((nglt,nglt))
Ford = np.zeros((nglt))
dord = np.zeros((nglt))
for i in range(nglt):
    Ford[i] = F[ordem[i]]
    dord[i] = d[ordem[i]]
    for j in range(nglt):
        Kgord[i,j] = Kg[ordem[i],ordem[j]]

# Sub matriz de rigidez para solução do problema estático (K u = F) com inclusão das condições de contorno
K = Kgord[:dim,:dim]
Fl = Ford[:dim]

# Solução estática
dl = linalg.solve(K,Fl - np.dot(Kgord[:dim,dim:],dord[dim:]))
for i in range(dim):
    d[ordem[i]] = dl[i]
print (f'Deslocamentos: {d}')
Ux = np.zeros((nnos)) # Deslocamentos na direção x
Uy = np.zeros((nnos)) # Deslocamentos na direção y
for i in range(nnos):
    Ux[i] = d[2*i]
    Uy[i] = d[2*i+1]

# Criando um indexador de nós x elementos
index_nos = []
for i in range(nnos):
    index_nos.append([])

# Cálculo das tensões para elementos T3
print('Tensões nos elementos:')
list_S = []
for i in range(nelem):
    posel = list_pos[i]
    ue = np.array([d[posel[0]],d[posel[1]],d[posel[2]],d[posel[3]],d[posel[4]],d[posel[5]]])
    D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
    B = list_B[i]
    S = np.dot(D,np.dot(B,ue))
    list_S.append(S)
    print(f'Elemento {i}:')
    print(f'Sx = {S[0]}')
    print(f'Sy = {S[1]}')
    print(f'Sxy = {S[2]}')
    for j in range(Inci.shape[1]-1):
        index_nos[Inci[i,j+1]].append(i)
# Tensões médias nos nós para elemento T3
Sx = np.zeros((nnos))
Sy = np.zeros((nnos))
Sxy = np.zeros((nnos))
for i in range(nnos):
    for j in index_nos[i]:
        S += list_S[j]
    S = 1./len(index_nos[i]) * S
    Sx[i] = S[0]
    Sy[i] = S[1]
    Sxy[i] = S[2]
print(f'Sx = {Sx}')
print(f'Sy = {Sy}')
print(f'Sxy = {Sxy}')

# Plotagem da estrutura deformada
pltmef.plotdef2D(Coord,Inci,d)

# Plotagem dos deslocamentos como gráficos de contorno
pltmef.plot_contor2D(tpel,Coord,Inci,Ux,'Ux')
pltmef.plot_contor2D(tpel,Coord,Inci,Uy,'Uy')

# Plotagem das tensões como gráficos de contorno
pltmef.plot_contor2D(tpel,Coord,Inci,Sx,'Sx')
pltmef.plot_contor2D(tpel,Coord,Inci,Sy,'Sy')
pltmef.plot_contor2D(tpel,Coord,Inci,Sxy,'Sxy')