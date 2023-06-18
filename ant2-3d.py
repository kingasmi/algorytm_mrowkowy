import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def rastrigin_function(x, y):
    """Funkcja celu - Rastrigin"""
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

class AntColony:
    def __init__(self, num_ants, bounds, alpha, beta, evaporation_rate):
        self.num_ants = num_ants
        self.bounds = bounds
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.best_solution = None
        self.best_fitness = float('inf')
        self.initialize_ants()

    def initialize_ants(self):
        self.ants = []
        for _ in range(self.num_ants):
            ant = {'x': np.random.uniform(*self.bounds),
                   'y': np.random.uniform(*self.bounds),
                   'fitness': None}
            self.ants.append(ant)

    def update_ant_fitness(self):
        for ant in self.ants:
            x, y = ant['x'], ant['y']
            ant['fitness'] = rastrigin_function(x, y)
            if ant['fitness'] < self.best_fitness:
                self.best_solution = (x, y)
                self.best_fitness = ant['fitness']

    def update_pheromone(self):
        for ant in self.ants:
            x, y = ant['x'], ant['y']
            pheromone = self.evaporation_rate * rastrigin_function(x, y)
            ant['x'] += np.random.uniform(-self.alpha, self.alpha) * pheromone
            ant['y'] += np.random.uniform(-self.alpha, self.alpha) * pheromone

    def optimize(self, num_iterations):
        self.initialize_ants()
        self.update_ant_fitness()
        history = [(ant['x'], ant['y'], ant['fitness']) for ant in self.ants]
        for _ in range(num_iterations):
            self.update_pheromone()
            self.update_ant_fitness()
            history.extend([(ant['x'], ant['y'], ant['fitness']) for ant in self.ants])
        return history

# Parametry algorytmu mrówkowego
num_ants = 10
bounds = (-5, 5)  # Zakres dla x i y
alpha = 0.1
beta = 0.1
evaporation_rate = 0.5
num_iterations = 50

# Inicjalizacja algorytmu
ant_colony = AntColony(num_ants, bounds, alpha, beta, evaporation_rate)

# Optymalizacja
history = ant_colony.optimize(num_iterations)

# Tworzenie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(*bounds, 100)
Y = np.linspace(*bounds, 100)
X, Y = np.meshgrid(X, Y)
Z = rastrigin_function(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Animacja ścieżek mrówek
scat = ax.scatter([], [], [], c='red', marker='o')

def update(frame):
    if frame < len(history):
        x, y, _ = zip(*history[:frame])
    else:
        x, y, _ = [], [], []
    scat._offsets3d = (x, y, rastrigin_function(x, y))
    return scat,

ani = FuncAnimation(fig, update, frames=len(history), blit=True)

# Wyświetlenie animacji
plt.show()
