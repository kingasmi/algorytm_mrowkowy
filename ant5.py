import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

n_ants = 100
n_iterations = 100
step_size = 0.1

ants = np.random.uniform(-10, 10, (n_ants, 2))
best_ant = None
best_score = np.inf

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Wykres 2D dla funkcji celu (sfera)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = sphere_function(X, Y)

ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
sc = ax1.scatter(ants[:, 0], ants[:, 1])
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_title('Funkcja celu - sfera')

def update(frame):
    global ants, best_ant, best_score
    direction = np.random.uniform(-1, 1, (n_ants, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ants += step_size * direction
    ants = np.clip(ants, -10, 10)
    scores = sphere_function(ants[:, 0], ants[:, 1])
    best_ant = ants[np.argmin(scores)]
    best_score = np.min(scores)
    sc.set_offsets(ants)
    ax2.clear()
    ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax2.scatter(ants[:, 0], ants[:, 1])
    ax2.scatter(best_ant[0], best_ant[1], color='red')
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_title(f'Iteracja: {frame+1}\nNajlepszy wynik: {best_score:.2f}')

ani = FuncAnimation(fig, update, frames=n_iterations, interval=200, repeat=False)
plt.show()
