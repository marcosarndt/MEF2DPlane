# Método dos Elementos Finitos para Problemas de Estado plano de tensão e deformação

import numpy as np
from scipy import linalg
import marcha_tempolin as mtemp
import plotmef2d as pltmef
import ler_dados
import integ

# Matriz Constitutiva
def constitutiva(tpproblema,E,ni):
    if tpproblema == 'EPT':
        De = E/(1-ni**2) * np.array([[1,ni,0],[ni,1,0],[0,0,(1-ni)/2]])
    else:
        De = E/((1+ni)*(1-2*ni)) * np.array([[1-ni,ni,0],[ni,1-ni,0],[0,0,(1-2*ni)/2]])
    return(De)

# Matriz de rigidez para elementos T3
def rigT3():
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
    return(Kg,list_B,list_pos)

# Matriz de massa para elementos T3
def massT3():
    for i in range(nelem):
        ro = Mat[Inci[i,0],3]
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        Ae = 1/2.*(Coord[n2,0]*Coord[n3,1]+Coord[n1,0]*Coord[n2,1]+Coord[n3,0]*Coord[n1,1]-Coord[n2,0]*Coord[n1,1]
                   -Coord[n1,0]*Coord[n3,1]-Coord[n3,0]*Coord[n2,1])
        if dados_analise[0] == 'EPT':
            t = Mat[Inci[i,0],0]
        else:
            t = 1.
        Maux = np.array([[2.,0.,1.,0.,1.,0.],
                         [0.,2.,0.,1.,0.,1.],
                         [1.,0.,2.,0.,1.,0.],
                         [0.,1.,0.,2.,0.,1.],
                         [1.,0.,1.,0.,2.,0.],
                         [0.,1.,0.,1.,0.,2.]])
        Me = (t*Ae*ro/12)*Maux
        pos = list_pos[i]
        for j in range(6):
            for k in range(6):
                Mg[pos[j],pos[k]] += Me[j,k]
    return(Mg)

# Matriz de rigidez para elementos Q4 isoparamétricos: quadratura de Gauss com 2 pontos em cada direção
def rigQ4(n): # n é o número de pontos de Gauss em cada direção
    ptg, w = integ.gauss_quad_2d(n)
    nptg = n * n
    for i in range(nelem):
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        if dados_analise[0] == 'EPT':
            t = Mat[Inci[i,0],0]
        else:
            t = 1.
        Ke = np.zeros((8,8))
        for j in range(nptg):
            xi = ptg[j,0]
            eta = ptg[j,1]
            dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
            J = np.dot(dxi,cnos)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            invJ = 1/detJ * np.array([[J[1,1],-J[0,1]],
                                      [-J[1,0],J[0,0]]])
            dx = np.dot(invJ,dxi)
            B = np.array([[dx[0,0],0,dx[0,1],0,dx[0,2],0,dx[0,3],0],
                          [0,dx[1,0],0,dx[1,1],0,dx[1,2],0,dx[1,3]],
                          [dx[1,0],dx[0,0],dx[1,1],dx[0,1],dx[1,2],dx[0,2],dx[1,3],dx[0,3]]])
            F = t * detJ * np.dot(np.transpose(B),np.dot(D,B))
            Ke += w[j]*F
        pos = [2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1,2*n4,2*n4+1]
        list_pos.append(pos)
        for j in range(8):
            for k in range(8):
                Kg[pos[j],pos[k]] += Ke[j,k]
    return(Kg,list_pos)

# Matriz de massa para elementos Q4 isoparamétricos: quadratura de Gauss com 2 pontos em cada direção
def massQ4(n): # n x n pontos de Gauss
    ptg, w = integ.gauss_quad_2d(n)
    nptg = n * n
    for i in range(nelem):
        ro = Mat[Inci[i,0],3]
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        if dados_analise[0] == 'EPT':
            t = Mat[Inci[i,0],0]
        else:
            t = 1.
        Me = np.zeros((8,8))
        for j in range(nptg):
            xi = ptg[j,0]
            eta = ptg[j,1]
            dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
            J = np.dot(dxi,cnos)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            Nv = [1./4.*(1+xi)*(1-eta), 1./4.*(1+xi)*(1+eta), 1./4.*(1-xi)*(1+eta), 1./4.*(1-xi)*(1-eta)]
            N = np.array([[Nv[0],0.,Nv[1],0.,Nv[2],0.,Nv[3],0.],
                          [0.,Nv[0],0.,Nv[1],0.,Nv[2],0.,Nv[3]]])
            F = t * ro * detJ * np.dot(np.transpose(N),N)
            Me += w[j]*F
        pos = list_pos[i]
        for j in range(8):
            for k in range(8):
                Mg[pos[j],pos[k]] += Me[j,k]
    return(Mg)

# Cálculo das tensões nodais nos elementos CST (T3) (média dos elementos que possuem o nó)
def stressT3():
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
        S = np.zeros((3))
        for j in index_nos[i]:
            S += list_S[j]
        S = 1./len(index_nos[i]) * S
        Sx[i] = S[0]
        Sy[i] = S[1]
        Sxy[i] = S[2]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Cálculo das tensões nodais recuperadas nos elementos CST (T3) pela técnica de recuperação superconvergente de
# Zienkiewicz-Zhu (Mendonça, Fancello, 2019)
def stressT3ZZ():
    print('Tensões nos elementos:')
    list_S = []
    coord_cg = []
    for i in range(nelem):
        xg = 0
        yg = 0
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
            xg += Coord[Inci[i,j+1],0]
            yg += Coord[Inci[i,j+1],1]
        coord_cg.append([float(xg/3.),float(yg/3.)])
    # Tensões recuperadas nos nós para elemento T3
    Sx = np.zeros((nnos))
    Sy = np.zeros((nnos))
    Sxy = np.zeros((nnos))
    for i in range(nnos):
        Fx = np.zeros((3,1))
        Fy = np.zeros((3,1))
        Fxy = np.zeros((3,1))
        Ak = np.zeros((3,3))
        for j in index_nos[i]:
            x = coord_cg[j][0]
            y = coord_cg[j][1]
            p = np.array([[1], [x], [y]])
            Ak += np.dot(p,np.transpose(p))
            Fx += list_S[j][0] * p
            Fy += list_S[j][1] * p
            Fxy += list_S[j][2] * p
        Sk = linalg.solve(Ak,Fx)
        Sx[i] = Sk[0] + Coord[i,0] * Sk[1] + Coord[i,1] * Sk[2]
        Sk = linalg.solve(Ak,Fy)
        Sy[i] = Sk[0] + Coord[i,0] * Sk[1] + Coord[i,1] * Sk[2]
        Sk = linalg.solve(Ak,Fxy)
        Sxy[i] = Sk[0] + Coord[i,0] * Sk[1] + Coord[i,1] * Sk[2]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Cálculo das tensões nodais nos elementos Q4 isoparamétricos com recuperação de tensões no elemento
def stressQ4re():
    print('Tensões nos pontos de Gauss dos elementos:')
    Sn = np.zeros((nnos,3))
    rep = np.zeros((nnos))
    # pontos de Gauss
    nptg = 4
    ptg = [[1./(3.**0.5),-1./(3.**0.5)],[1./(3.**0.5),1./(3.**0.5)],[-1./(3.**0.5),1./(3.**0.5)],[-1./(3.**0.5),-1./(3.**0.5)]]
    # nós extrapolados em relação aos pontos de Gauss
    noext = [[(3.**0.5),-(3.**0.5)],[(3.**0.5),(3.**0.5)],[-(3.**0.5),(3.**0.5)],[-(3.**0.5),-(3.**0.5)]]
    for i in range(nelem):
        posel = list_pos[i]
        ue = np.array([d[posel[0]],d[posel[1]],d[posel[2]],d[posel[3]],d[posel[4]],d[posel[5]],d[posel[6]],d[posel[7]]])
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        Sepg = np.zeros((3,nptg))
        # Tensões nos pontos de Gauss do elemento
        for j in range(nptg):
            xi = ptg[j][0]
            eta = ptg[j][1]
            dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
            J = np.dot(dxi,cnos)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            invJ = 1/detJ * np.array([[J[1,1],-J[0,1]],
                                      [-J[1,0],J[0,0]]])
            dx = np.dot(invJ,dxi)
            B = np.array([[dx[0,0],0,dx[0,1],0,dx[0,2],0,dx[0,3],0],
                          [0,dx[1,0],0,dx[1,1],0,dx[1,2],0,dx[1,3]],
                          [dx[1,0],dx[0,0],dx[1,1],dx[0,1],dx[1,2],dx[0,2],dx[1,3],dx[0,3]]])
            Spg = np.dot(D,np.dot(B,ue))
            Sepg[0,j] = Spg[0]
            Sepg[1,j] = Spg[1]
            Sepg[2,j] = Spg[2]
            print(f'Elemento {i} - ponto de Gauss ({xi,eta}):')
            print(f'Sx = {Spg[0]}')
            print(f'Sy = {Spg[1]}')
            print(f'Sxy = {Spg[2]}')
        # Recuperação das tensões nodais (Capítulo 28 – Stress Recovery. Notas de aula da disciplina Introduction to Finite 
        # Element Methods (ASEN 5007), Department of Aerospace Engineering Sciences, University of Colorado at Boulder. 
        # https://quickfem.com/theory/finite-element-analysis/
        for j in range(4):
            xi = noext[j][0]
            eta = noext[j][1]
            N = np.array([1./4.*(1+xi)*(1-eta),1./4.*(1+xi)*(1+eta),1./4.*(1-xi)*(1+eta),1./4.*(1-xi)*(1-eta)])
            S = np.dot(Sepg,N)
            print(f'Elemento {i} - Nó ({j}):')
            print(f'Sx = {S[0]}')
            print(f'Sy = {S[1]}')
            print(f'Sxy = {S[2]}')
            Sn[Inci[i,j+1],0] += S[0]
            Sn[Inci[i,j+1],1] += S[1]
            Sn[Inci[i,j+1],2] += S[2]
            rep[Inci[i,j+1]] +=1
    # Tensões médias recuperadas nos nós para elemento Q4
    Sx = np.zeros((nnos))
    Sy = np.zeros((nnos))
    Sxy = np.zeros((nnos))
    for i in range(nnos):
       Sx = 1./rep[i] * Sn[:,0]
       Sy = 1./rep[i] * Sn[:,1]
       Sxy = 1./rep[i] * Sn[:,2]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Cálculo das tensões nodais médias nos elementos Q4 isoparamétricos
def stressQ4():
    print('Tensões nos nós dos elementos:')
    Sn = np.zeros((nnos,3))
    rep = np.zeros((nnos))
    # coordenadas dos nós no sistema de coordenadas local
    ptg = [[1.,-1.],[1.,1.],[-1.,1.],[-1.,-1.]]
    for i in range(nelem):
        posel = list_pos[i]
        ue = np.array([d[posel[0]],d[posel[1]],d[posel[2]],d[posel[3]],d[posel[4]],d[posel[5]],d[posel[6]],d[posel[7]]])
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        # Tensões nos nós do elemento
        for j in range(4):
            xi = ptg[j][0]
            eta = ptg[j][1]
            dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
            J = np.dot(dxi,cnos)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            invJ = 1/detJ * np.array([[J[1,1],-J[0,1]],
                                      [-J[1,0],J[0,0]]])
            dx = np.dot(invJ,dxi)
            B = np.array([[dx[0,0],0,dx[0,1],0,dx[0,2],0,dx[0,3],0],
                          [0,dx[1,0],0,dx[1,1],0,dx[1,2],0,dx[1,3]],
                          [dx[1,0],dx[0,0],dx[1,1],dx[0,1],dx[1,2],dx[0,2],dx[1,3],dx[0,3]]])
            Spg = np.dot(D,np.dot(B,ue))
            print(f'Elemento {i} - Nó ({j}):')
            print(f'Sx = {Spg[0]}')
            print(f'Sy = {Spg[1]}')
            print(f'Sxy = {Spg[2]}')
            Sn[Inci[i,j+1],0] += Spg[0]
            Sn[Inci[i,j+1],1] += Spg[1]
            Sn[Inci[i,j+1],2] += Spg[2]
            rep[Inci[i,j+1]] +=1
    # Tensões médias nos nós para elemento Q4
    Sx = np.zeros((nnos))
    Sy = np.zeros((nnos))
    Sxy = np.zeros((nnos))
    for i in range(nnos):
       Sx = 1./rep[i] * Sn[:,0]
       Sy = 1./rep[i] * Sn[:,1]
       Sxy = 1./rep[i] * Sn[:,2]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Cálculo das tensões nodais recuperadas nos elementos Q4 isoparamétricos pela técnica de recuperação discreta superconvergente de
# Zienkiewicz-Zhu com 4 termos (Mendonça, Fancello, 2019)
def stressQ4ZZ():
    print('Tensões nos centroides paramétricos dos elementos:')
    list_S = []
    coord_cg = []
    for i in range(nelem):
        posel = list_pos[i]
        ue = np.array([d[posel[0]],d[posel[1]],d[posel[2]],d[posel[3]],d[posel[4]],d[posel[5]],d[posel[6]],d[posel[7]]])
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        # Tensões no centroide do elemento Q4
        xi = 0.
        eta = 0.
        dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
        J = np.dot(dxi,cnos)
        detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        invJ = 1/detJ * np.array([[J[1,1],-J[0,1]],
                                [-J[1,0],J[0,0]]])
        dx = np.dot(invJ,dxi)
        B = np.array([[dx[0,0],0,dx[0,1],0,dx[0,2],0,dx[0,3],0],
                    [0,dx[1,0],0,dx[1,1],0,dx[1,2],0,dx[1,3]],
                    [dx[1,0],dx[0,0],dx[1,1],dx[0,1],dx[1,2],dx[0,2],dx[1,3],dx[0,3]]])
        S = np.dot(D,np.dot(B,ue))
        list_S.append(S)
        print(f'Elemento {i} - centróide paramétrico (0,0):')
        print(f'Sx = {S[0]}')
        print(f'Sy = {S[1]}')
        print(f'Sxy = {S[2]}')
        coord_cg.append([float(Coord[n1,0]+Coord[n2,0]+Coord[n3,0]+Coord[n4,0])/4.,
                         float(Coord[n1,1]+Coord[n2,1]+Coord[n3,1]+Coord[n4,1])/4.])
        for j in range(Inci.shape[1]-1):
            index_nos[Inci[i,j+1]].append(i)
    # Tensões recuperadas nos nós para elemento Q4
    Sx = np.zeros((nnos))
    Sy = np.zeros((nnos))
    Sxy = np.zeros((nnos))
    for i in range(nnos):
        Fx = np.zeros((4,1))
        Fy = np.zeros((4,1))
        Fxy = np.zeros((4,1))
        Ak = np.zeros((4,4))
        # Se for um nó do contorno considera o patch do nó interior correspondente
        if len(index_nos[i]) <= 2:
            for k in range(4):
                node = Inci[index_nos[i][0],k+1]
                if set(index_nos[i]).issubset(index_nos[node]) and len(index_nos[node]) > 2:
                    no_int = node
            index_nos[i] = index_nos[no_int]
        for j in index_nos[i]:
            x = coord_cg[j][0]
            y = coord_cg[j][1]
            p = np.array([[1], [x], [y], [x*y]])
            Ak += np.dot(p,np.transpose(p))
            Fx += list_S[j][0] * p
            Fy += list_S[j][1] * p
            Fxy += list_S[j][2] * p
        Sk = linalg.solve(Ak,Fx)
        Sx[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0] + Coord[i,0] * Coord[i,1] * Sk[3,0]
        Sk = linalg.solve(Ak,Fy)
        Sy[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0] + Coord[i,0] * Coord[i,1] * Sk[3,0]
        Sk = linalg.solve(Ak,Fxy)
        Sxy[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0] + Coord[i,0] * Coord[i,1] * Sk[3,0]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Cálculo das tensões nodais recuperadas nos elementos Q4 isoparamétricos pela técnica de recuperação discreta superconvergente de
# Zienkiewicz-Zhu (Mendonça, Fancello, 2019) com 3 termos lineares (1, x, y)
def stressQ4ZZ2():
    print('Tensões nos centroides paramétricos dos elementos:')
    list_S = []
    coord_cg = []
    for i in range(nelem):
        posel = list_pos[i]
        ue = np.array([d[posel[0]],d[posel[1]],d[posel[2]],d[posel[3]],d[posel[4]],d[posel[5]],d[posel[6]],d[posel[7]]])
        D = constitutiva(dados_analise[0],Mat[Inci[i,0],1],Mat[Inci[i,0],2])
        n1 = Inci[i,1]
        n2 = Inci[i,2]
        n3 = Inci[i,3]
        n4 = Inci[i,4]
        cnos = np.array([[Coord[n1,0],Coord[n1,1]],
                          [Coord[n2,0],Coord[n2,1]],
                          [Coord[n3,0],Coord[n3,1]],
                          [Coord[n4,0],Coord[n4,1]]])
        # Tensões no centroide do elemento Q4
        xi = 0.
        eta = 0.
        dxi = np.array([[1./4.*(1-eta), 1./4.*(1+eta), -1./4.*(1+eta), -1./4.*(1-eta)],
                         [-1./4.*(1+xi), 1./4.*(1+xi), 1./4.*(1-xi), -1./4.*(1-xi)]])
        J = np.dot(dxi,cnos)
        detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        invJ = 1/detJ * np.array([[J[1,1],-J[0,1]],
                                [-J[1,0],J[0,0]]])
        dx = np.dot(invJ,dxi)
        B = np.array([[dx[0,0],0,dx[0,1],0,dx[0,2],0,dx[0,3],0],
                    [0,dx[1,0],0,dx[1,1],0,dx[1,2],0,dx[1,3]],
                    [dx[1,0],dx[0,0],dx[1,1],dx[0,1],dx[1,2],dx[0,2],dx[1,3],dx[0,3]]])
        S = np.dot(D,np.dot(B,ue))
        list_S.append(S)
        print(f'Elemento {i} - centróide paramétrico (0,0):')
        print(f'Sx = {S[0]}')
        print(f'Sy = {S[1]}')
        print(f'Sxy = {S[2]}')
        coord_cg.append([float(Coord[n1,0]+Coord[n2,0]+Coord[n3,0]+Coord[n4,0])/4.,
                         float(Coord[n1,1]+Coord[n2,1]+Coord[n3,1]+Coord[n4,1])/4.])
        for j in range(Inci.shape[1]-1):
            index_nos[Inci[i,j+1]].append(i)
    # Tensões recuperadas nos nós para elemento Q4
    Sx = np.zeros((nnos))
    Sy = np.zeros((nnos))
    Sxy = np.zeros((nnos))
    for i in range(nnos):
        Fx = np.zeros((3,1))
        Fy = np.zeros((3,1))
        Fxy = np.zeros((3,1))
        Ak = np.zeros((3,3))
        # Se for um nó do contorno considera o patch do nó interior correspondente
        if len(index_nos[i]) <= 2:
            for k in range(4):
                node = Inci[index_nos[i][0],k+1]
                if set(index_nos[i]).issubset(index_nos[node]) and len(index_nos[node]) > 2:
                    no_int = node
            index_nos[i] = index_nos[no_int]
        for j in index_nos[i]:
            x = coord_cg[j][0]
            y = coord_cg[j][1]
            p = np.array([[1], [x], [y]])
            Ak += np.dot(p,np.transpose(p))
            Fx += list_S[j][0] * p
            Fy += list_S[j][1] * p
            Fxy += list_S[j][2] * p
        Sk = linalg.solve(Ak,Fx)
        Sx[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0]
        Sk = linalg.solve(Ak,Fy)
        Sy[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0]
        Sk = linalg.solve(Ak,Fxy)
        Sxy[i] = Sk[0,0] + Coord[i,0] * Sk[1,0] + Coord[i,1] * Sk[2,0]
    print(f'Sx = {Sx}')
    print(f'Sy = {Sy}')
    print(f'Sxy = {Sxy}')
    return(Sx,Sy,Sxy)

# Programa Principal
# Entrada de dados
# Tipo de entrada de dados
string = input('Qual o caminho e o nome do arquivo de modelo que você deseja abrir (nome do arquivo sem extensão? ')
# Transforma a string de entrada em uma raw string
arquivo = string.replace("\\", "\\\\")

# Leitura de dados da malha Gmsh
# Inserir as linhas no GMSH em sentido anti-horário para que a incidência fornceça os nós nesse sentido
Coord,Inci,tpel,list_mat,dict_cc = ler_dados.read_mesh2D(arquivo)
nnos = Coord.shape[0] # número de nós
nelem = Inci.shape[0] # número de elementos
nglt = 2*nnos # número total de graus de liberdade

# Plotando a malha 
resp1 = input('Deseja plotar a malha? (S/N) ')
if resp1 == 'S' or resp1 == 's':
    resp2 = input('Deseja mostrar os números dos nós? (S/N) ')
    if resp2 == 'S' or resp2 == 's':
        pltmef.plot_malha2D(Coord,Inci,True)
    else:
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

# Matrizes de rigidez e massa elementares e global
list_B = []
list_pos = []
Kg = np.zeros((nglt,nglt))
Kgord = np.zeros((nglt,nglt))
if tpel == 'triangle':
    Kg,list_B,list_pos = rigT3()
else:
    if tpel == 'quad':
        Kg,list_pos = rigQ4(2)
# print(list_B)
# print(list_pos)
print(f'Matriz de rigidez global: {Kg}')
if (dados_analise[1] == 'modal') or (dados_analise[1] == 'transiente'):
    Mg = np.zeros((nglt,nglt))
    Mgord = np.zeros((nglt,nglt))
    if tpel == 'triangle':
        Mg = massT3()
    else:
        if tpel == 'quad':
            Mg = massQ4(2)

# Reordenação da matriz de rigidez, massa e vetor de forças
Ford = np.zeros((nglt))
dord = np.zeros((nglt))
for i in range(nglt):
    Ford[i] = F[ordem[i]]
    dord[i] = d[ordem[i]]
    for j in range(nglt):
        Kgord[i,j] = Kg[ordem[i],ordem[j]]
        if (dados_analise[1] == 'modal') or (dados_analise[1] == 'transiente'):
            Mgord[i,j] = Mg[ordem[i],ordem[j]]

# Sub matriz de rigidez e massa para solução do problema estático (K u = F) com inclusão das condições de contorno
K = Kgord[:dim,:dim]
print(f'K: {K}')
if (dados_analise[1] == 'modal') or (dados_analise[1] == 'transiente'):
    M = Mgord[:dim,:dim]
    print(f'M: {M}')
Fl = Ford[:dim]
print(Fl)

if dados_analise[1] == 'estatica':
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

    # Plotagem da estrutura deformada
    resp1 = input('Deseja plotar a estrutura deformada? (S/N) ')
    if resp1 == 's' or resp1 == 'S':
        resp2 = input('Deseja usar slide [digite 1] ou escala fixa [digite escala > 1.0]? ')
        escala = float(resp2)
        if escala > 1.0:
            pltmef.plotdef2DX(Coord,Inci,d,escala)
        else:
            pltmef.plotdef2D(Coord,Inci,d)

    # Plotagem dos deslocamentos como gráficos de contorno
    pltmef.plot_contor2D(tpel,Coord,Inci,Ux,'Ux')
    pltmef.plot_contor2D(tpel,Coord,Inci,Uy,'Uy')

    # Criando um indexador de nós x elementos
    index_nos = []
    for i in range(nnos):
        index_nos.append([])

    # Cálculo das tensões para elementos T3 e Q4
    resp1 = input('Técnica de recuperação de tensões: [1] Média, [2] ZZ ou [3] Boulder (somente Q4): ')
    resp2 = 'N'
    match resp1:
        case '1':
            if tpel == 'triangle':
                Sx,Sy,Sxy = stressT3()
            else:
                if tpel == 'quad':
                    Sx,Sy,Sxy = stressQ4()
            resp2 = 'S'
        case '2':
            if tpel == 'triangle':
                Sx,Sy,Sxy = stressT3ZZ()
            else:
                if tpel == 'quad':
                    Sx,Sy,Sxy = stressQ4ZZ()
            resp2 = 'S'
        case '3':
            if tpel == 'triangle':
                Sx,Sy,Sxy = stressT3()
            else:
                if tpel == 'quad':
                    Sx,Sy,Sxy = stressQ4re()
            resp2 = 'S'
    if resp2 == 'S':
        # Plotagem das tensões como gráficos de contorno
        pltmef.plot_contor2D(tpel,Coord,Inci,Sx,'Sx')
        pltmef.plot_contor2D(tpel,Coord,Inci,Sy,'Sy')
        pltmef.plot_contor2D(tpel,Coord,Inci,Sxy,'Sxy')

if dados_analise[1] == 'modal' or (dados_analise[1] == 'transiente' and float(dados_analise[7]) > 0):
    # Solução modal
    autovalores, autovetores = linalg.eigh(K, M)
    freq = autovalores ** (1/2)
    print(f'Frequências: {freq}')
    # Normalização dos modos em relação à massa
    Mmodal = np.array(autovetores)
    modo = []
    for i in range(dim):
        Mn = np.dot(np.transpose(Mmodal[:,i]),np.dot(M,Mmodal[:,i]))
        modo.append(1/(Mn**(1/2)) * Mmodal[:,i])
        # print(f'Modo {i}: {modo[i]}')
    resp = 'S'
    while resp == 's' or resp == 'S':
        resp = input('Deseja plotar algum modo de vibração da estrutura (S ou N)? ')
        if resp == 's' or resp == 'S':
            dmodo = np.zeros((nglt))
            nmodo = int(input(f'Modo que deseja plotar (0 a {dim-1}):'))
            for i in range(dim):
                dmodo[ordem[i]] = modo[nmodo][i]
            pltmef.plotdef2D(Coord,Inci,dmodo)
            # Plotagem dos modos como gráficos de contorno
            Ux = np.zeros((nnos)) # Deslocamentos na direção x
            Uy = np.zeros((nnos)) # Deslocamentos na direção y
            for i in range(nnos):
                Ux[i] = dmodo[2*i]
                Uy[i] = dmodo[2*i+1]
            pltmef.plot_contor2D(tpel,Coord,Inci,Ux,'Ux')
            pltmef.plot_contor2D(tpel,Coord,Inci,Uy,'Uy')

if dados_analise[1] == 'transiente':
    # Solução transiente partindo do repouso
    txam = float(dados_analise[7])
    met_int = dados_analise[2]
    tp_force = dados_analise[3]
    arg_force = float(dados_analise[4])
    delta_t = float(dados_analise[5])
    tempo_final = float(dados_analise[6])
    if txam > 0:
        omega1 = freq[0]
        if dim == 1:
            omega2 = freq[0]
        else:
            omega2 = freq[1]
        a0 = txam*2*omega1*omega2/(omega1+omega2)
        a1 = txam*2/(omega1+omega2)
        C = a0 * M + a1 * K
    else:
        C = np.zeros((dim,dim))
    u0 = np.zeros((dim)) # repouso
    up0 = np.zeros((dim)) # repouso
    mtemp.marcha_tempo(met_int,K,M,C,Fl,u0,up0,tp_force,arg_force,delta_t,tempo_final,arquivo)
    resp = 'S'
    while resp == 's' or resp == 'S':
        resp = input('Deseja plotar a resposta transiente da estrutura (S ou N)? ')
        if resp == 's' or resp == 'S':
            var = input(f'Qual variável (deslocamento, velocidade ou aceleracao)? ')
            ntran = int(input(f'Qual o nó (0 a {nnos-1}): '))
            dir = int(input(f'Qual a direção (0 para x ou 1 para y): '))
            gld = 2 * ntran + dir
            try:
                # Encontrando o índice do valor 
                print(ordem)
                indice = ordem.index(gld)
                print(indice)
                pltmef.plot_transi(arquivo,indice,var,met_int)
            except ValueError: 
                print(f'O grau de liberdade é restrito.')