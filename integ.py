import numpy as np

# Quadratura Gaussiana para quadrados (-1,1)
def gauss_quad_2d(n): # n x n pontos de integração
    # pontos e pesos 1D
    x, w = np.polynomial.legendre.leggauss(n)
    # produto cartesiano
    X, Y = np.meshgrid(x, x)
    WX, WY = np.meshgrid(w, w)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    ws  = (WX * WY).ravel()
    return pts, ws

# Quadratura Gaussiana para triângulos
def gauss_triang_2d (n): # n : número de pontos
    if n == 1:
        pts = np.array([[1/3, 1/3]])
        ws  = np.array([1.0])
    if n == 3:
        pts = np.array([
            [2/3, 1/6],
            [1/6, 2/3],
            [1/6, 1/6]])
        ws = np.array([1/3, 1/3, 1/3])
    if n == 4:
        pts = np.array([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2]])
        ws = np.array([-27/48, 25/48, 25/48, 25/48])
    if n == 6:
        a = 0.445948490915965
        b = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        pts = np.array([
            [a, a],
            [a, 1-2*a],
            [1-2*a, a],
            [b, b],
            [b, 1-2*b],
            [1-2*b, b]])
        ws = np.array([w1, w1, w1, w2, w2, w2])
    if n == 7:
        pts = np.array([
            [1/3, 1/3],
            [0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.059715871789770, 0.470142064105115],
            [0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.797426985353087, 0.101286507323456]])
        ws = np.array([
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827])
    return pts, ws