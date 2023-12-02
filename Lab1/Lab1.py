import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

t = sp.Symbol('t')

r = 2+sp.cos(6*t) # начальные уравнения
phi = t+1.2*sp.cos(6*t)
x = r*sp.cos(phi)
y = r*sp.sin(phi)

Vx = sp.diff(x, t)  # ур-е скорости через дифф-ие
Vy = sp.diff(y, t)
a_x = sp.diff(Vx, t)    # ур-е ускорения через дифф-ие
a_y = sp.diff(Vy, t)

T = np.linspace(0, 10, 1000)    # создаем послед-ть из 1000 точек на [0, 10]

X = np.zeros_like(T)    # массив нулей на основе T
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
A_x = np.zeros_like(T)
A_y = np.zeros_like(T)

for i in np.arange(len(T)):     # массив с эл-ми от 0 до len(T) [...)
    X[i] = sp.Subs(x, t, T[i])  # подтавляем в выр-ие x вместо t знач-ие T[i]
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    A_x[i] = sp.Subs(a_x, t, T[i])
    A_y[i] = sp.Subs(a_y, t, T[i])

fig = plt.figure()  # создали объект-контейнер

ax1 = fig.add_subplot(1, 1, 1)  # создали 1 график
ax1.axis('equal')   # стабилизация изображения при изменении масштаба
ax1.set(xlim=[-4, 4], ylim=[-4, 4])     # установка границ для окна с графиком

ax1.plot(X, Y)  # рисуем график по точкам из X и Y
#ax1.plot([X.min(), X.max()], [0, 0], 'black')

P, = ax1.plot(X[0], Y[0], marker='o')   # распаковка данных для итерируемых объектов / plot элемент
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r')
Aline, = ax1.plot([X[0], X[0]+A_x[0]], [Y[0], Y[0]+A_y[0]], 'blue')

ArrowX = np.array([-0.1, 0, -0.1])  # создание массива
ArrowY = np.array([0.1, 0, -0.1])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
RArrowX_A, RArrowY_A = Rot2D(ArrowX, ArrowY, math.atan2(A_y[i], A_x[i]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')
AArrow, = ax1.plot(RArrowX_A+X[0], RArrowY_A+Y[0], 'blue')

Rline, = ax1.plot([0, X[0]], [0, Y[0]], 'black')
RArrow, = ax1.plot(RArrowX_A + VX[0] + X[0], RArrowY_A + VY[0] + Y[0], 'black')

def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+X[i]+VX[i], RArrowY+Y[i]+VY[i])
    Aline.set_data([X[i], X[i]+A_x[i]], [Y[i], Y[i]+A_y[i]])
    RArrowX_A, RArrowY_A = Rot2D(ArrowX, ArrowY, math.atan2(A_y[i], A_x[i]))
    AArrow.set_data(RArrowX_A + X[i]+A_x[i], RArrowY_A + Y[i]+A_y[i])
    Rline.set_data([0, X[i]], [0, Y[i]])
    RArrowX_A, RArrowY_A = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RArrowX_A + X[i], RArrowY_A + Y[i])
    return P, VLine, VArrow, Aline, AArrow, Rline, RArrow

anim = FuncAnimation(fig, anima, frames=1000, interval=0.5, repeat=False)

plt.show()  # вывод всех фигур на экран