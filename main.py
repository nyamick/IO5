import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm

def Сonditional_F(x):
    return (x[0] ** 2) + (2 * x[1] ** 2) - (5 * x[0]) - (7 * x[1]) #целевая функция


def Graf_Сonditional_F(x):
    return (-x[0] ** 2) - (2 * x[1] ** 2) + (5 * x[0]) + (7 * x[1])  # целевая функция

def ff_der1(x):
    der = np.zeros_like(x)
    der[0] = 5 - 2 * x[0]
    der[1] = 7 - 4 * x[1]
    return der


ineq_cons = {'type': 'ineq',
             'fun': lambda x: np.array([-16 + 2 * x[0] + x[1]]),   #ограничения НЕравенствами
             'jac': lambda x: np.array([[2, 1]])
             }

eq_cons = {'type': 'eq',
           'fun': lambda x: np.array ([x [0] + x [1] - 7]),        #ограничения РАВЕНСТВАМИ
           'jac': lambda x: np.array ([1.0, 1.0])
          }

bnd = [(0, float("inf")),
       (0, float("inf"))]
x0 = np.array([0, 1])

opt = minimize(Сonditional_F, x0, method='SLSQP', jac=ff_der1,
               constraints=[ineq_cons, eq_cons],
               options={'ftol': 1e-9, 'disp': True},
               bounds=bnd)
print(opt)

fig = plt.figure(figsize=[15, 10])
ax = fig.add_subplot(projection='3d')

# Задаем угол обзора
ax.view_init(45, 30)

# определяем область отрисовки графика
X = np.arange(0, 15, 10)  #добавление графику ограничений
Y = np.arange(-20, 20, 1) #добавление графику ограничений
X, Y = np.meshgrid(X, Y)
Z = Graf_Сonditional_F(np.array([X, Y]))

# Рисуем поверхность
ax.plot_surface(X, Y, Z, cmap='coolwarm')
plt.title(r'Условная оптимизация', fontsize=16, y=1.05)

plt.show()