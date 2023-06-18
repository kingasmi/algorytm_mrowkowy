#caly kod programu 
# ant + ant6 
# wykresy w 2d - obie funkcje 
# wykresy liniowe - obie funkcje
# wykresy w 3d - funkcja celu - strefa

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    ax1.set_title(f'Funkcja celu - sfera\nIteracja: {frame+1}\nNajlepszy wynik: {best_score_sphere:.2f}')
    ax2.set_title(f'Funkcja celu - Rastrigin\nIteracja: {frame+1}\nNajlepszy wynik: {best_score_rastrigin:.2f}')

ani1 = FuncAnimation(fig1, update, frames=n_iterations, interval=200, repeat=False)
ani2 = FuncAnimation(fig2, update, frames=n_iterations, interval=200, repeat=False)

#ani1.save('sphere_animation.gif', writer='imagemagick')
#ani2.save('rastrigin_animation.gif', writer='imagemagick')

# Wykres zbieżności funkcji celu - sfera
best_scores_sphere = []
best_scores_rastrigin = []

for it in range(n_iterations):
    # Każda mrówka wykonuje ruch
    for i in range(n_ants):
        # Losowo wybiera kierunek
        direction = np.random.uniform(-1, 1, 2)
        direction /= np.linalg.norm(direction) # Normalize to unit vector

        # Zaktualizuj pozycje
        ants[i] += step_size * direction

        # Kontrola granicy
        ants[i] = np.clip(ants[i], -10, 10)

        # Zaktualizuj najlepsze rozwiązanie dla funkcji celu - sfera
        score_sphere = sphere_function(ants[i][0], ants[i][1])
        if score_sphere < best_score_sphere:
            best_score_sphere = score_sphere
            best_ant_sphere = ants[i].copy()

        # Zaktualizuj najlepsze rozwiązanie dla funkcji celu - Rastrigin
        score_rastrigin = rastrigin_function(ants[i][0], ants[i][1])
        if score_rastrigin < best_score_rastrigin:
            best_score_rastrigin = score_rastrigin
            best_ant_rastrigin = ants[i].copy()

    best_scores_sphere.append(best_score_sphere)
    best_scores_rastrigin.append(best_score_rastrigin)

# Wykres zbieżności funkcji celu - sfera
plt.figure()
plt.plot(best_scores_sphere)
plt.xlabel('Iteracja')
plt.ylabel('Najlepszy wynik')
plt.title('Zbieżność - Funkcja celu - sfera')
#plt.savefig('sphere_convergence.png')

# Wykres zbieżności funkcji celu - Rastrigin
plt.figure()
plt.plot(best_scores_rastrigin)
plt.xlabel('Iteracja')
plt.ylabel('Najlepszy wynik')
plt.title('Zbieżność - Funkcja celu - Rastrigin')
#plt.savefig('rastrigin_convergence.png')




# Wykres 3D dla funkcji celu - sfera
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z_sphere, cmap='viridis', alpha=0.8)
sc3 = ax3.scatter(ants[:, 0], ants[:, 1], sphere_function(ants[:, 0], ants[:, 1]), c='b')
best_sc3 = ax3.scatter([], [], [], c='r', label='Najlepszy wynik')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Funkcja celu - sfera')

best_score_text = ax3.text(0.95, 0.95, 0.95, '', transform=ax3.transAxes, ha='right', va='top')

def update_3d(frame):
    global ants, best_ant_sphere, best_score_sphere
    direction = np.random.uniform(-1, 1, (n_ants, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ants += step_size * direction
    ants = np.clip(ants, -10, 10)
    scores_sphere = sphere_function(ants[:, 0], ants[:, 1])
    best_ant_sphere = ants[np.argmin(scores_sphere)]
    best_score_sphere = np.min(scores_sphere)
    sc3._offsets3d = (ants[:, 0], ants[:, 1], scores_sphere)
    best_sc3._offsets3d = ([best_ant_sphere[0]], [best_ant_sphere[1]], [sphere_function(*best_ant_sphere)])
    best_score_text.set_text(f'Najlepszy wynik: {best_score_sphere:.2f}')
    ax3.set_title(f'Funkcja celu - sfera\nIteracja: {frame+1}')

ani3 = FuncAnimation(fig3, update_3d, frames=n_iterations, interval=200, repeat=False)

#ani3.save('sphere_animation_3d.gif', writer='imagemagick')

plt.show()