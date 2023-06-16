import numpy as np
import matplotlib.pyplot as plt

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

best_scores_sphere = []
best_scores_rastrigin = []
best_ant_sphere = None
best_ant_rastrigin = None
best_score_sphere = np.inf
best_score_rastrigin = np.inf

for it in range(n_iterations):
    # Each ant makes a move
    for i in range(n_ants):
        # Randomly select a direction
        direction = np.random.uniform(-1, 1, 2)
        direction /= np.linalg.norm(direction) # Normalize to unit vector

        # Update position
        ants[i] += step_size * direction

        # Boundary check
        ants[i] = np.clip(ants[i], -10, 10)

        # Update the best solution for sphere function
        score_sphere = sphere_function(ants[i][0], ants[i][1])
        if score_sphere < best_score_sphere:
            best_score_sphere = score_sphere
            best_ant_sphere = ants[i].copy()

        # Update the best solution for rastrigin function
        score_rastrigin = rastrigin_function(ants[i][0], ants[i][1])
        if score_rastrigin < best_score_rastrigin:
            best_score_rastrigin = score_rastrigin
            best_ant_rastrigin = ants[i].copy()

    best_scores_sphere.append(best_score_sphere)
    best_scores_rastrigin.append(best_score_rastrigin)

# Plotting the convergence of the Sphere function
plt.figure()
plt.plot(best_scores_sphere)
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.title('Convergence - Sphere Function')

# Plotting the convergence of the Rastrigin function
plt.figure()
plt.plot(best_scores_rastrigin)
plt.xlabel('Iteration')
plt.ylabel('Best Score')
plt.title('Convergence - Rastrigin Function')
