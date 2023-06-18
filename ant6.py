#wykresy w 2D dla obu funkcji
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

def rastrigin_function(x, y):
    """Funkcja celu - Rastrigin"""
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

n_ants = 100
n_iterations = 100
step_size = 0.1

ants = np.random.uniform(-10, 10, (n_ants, 2))
best_ant_sphere = None
best_score_sphere = np.inf
best_ant_rastrigin = None
best_score_rastrigin = np.inf

# Wykres 2D dla funkcji celu - sfera
fig1, ax1 = plt.subplots(figsize=(6, 6))
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z_sphere = sphere_function(X, Y)
ax1.contourf(X, Y, Z_sphere, levels=50, cmap='viridis')
sc1 = ax1.scatter(ants[:, 0], ants[:, 1])
best_sc1 = ax1.scatter([], [], color='red')
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_title('Funkcja celu - sfera')

# Wykres 2D dla funkcji celu - Rastrigin
fig2, ax2 = plt.subplots(figsize=(6, 6))
Z_rastrigin = rastrigin_function(X, Y)
ax2.contourf(X, Y, Z_rastrigin, levels=50, cmap='viridis')
sc2 = ax2.scatter(ants[:, 0], ants[:, 1])
best_sc2 = ax2.scatter([], [], color='red')
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.set_title('Funkcja celu - Rastrigin')

def update(frame):
    global ants, best_ant_sphere, best_score_sphere, best_ant_rastrigin, best_score_rastrigin
    direction = np.random.uniform(-1, 1, (n_ants, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ants += step_size * direction
    ants = np.clip(ants, -10, 10)
    scores_sphere = sphere_function(ants[:, 0], ants[:, 1])
    scores_rastrigin = rastrigin_function(ants[:, 0], ants[:, 1])
    best_ant_sphere = ants[np.argmin(scores_sphere)]
    best_score_sphere = np.min(scores_sphere)
    best_ant_rastrigin = ants[np.argmin(scores_rastrigin)]
    best_score_rastrigin = np.min(scores_rastrigin)
    sc1.set_offsets(ants)
    sc2.set_offsets(ants)
    best_sc1.set_offsets(best_ant_sphere)
    best_sc2.set_offsets(best_ant_rastrigin)
    ax1.set_title(f'Iteracja: {frame+1}\nNajlepszy wynik: {best_score_sphere:.2f}')
    ax2.set_title(f'Iteracja: {frame+1}\nNajlepszy wynik: {best_score_rastrigin:.2f}')

ani1 = FuncAnimation(fig1, update, frames=n_iterations, interval=200, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=n_iterations, interval=200, repeat=False)

ani1.save('sphere_animation.gif', writer='imagemagick')
ani2.save('rastrigin_animation.gif', writer='imagemagick')

plt.show()
