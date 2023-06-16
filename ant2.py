import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

class AntColony:
    def __init__(self, num_ants, num_nodes, distance_matrix, alpha=1, beta=2, evaporation_rate=0.5, pheromone_deposit=1):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.distance_matrix = distance_matrix  # Dodaj atrybut distance_matrix
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.pheromone_matrix = np.ones((num_nodes, num_nodes))

    def calculate_path_distance(self, path):
        distance = 0
        num_nodes = len(path)

        for i in range(num_nodes - 1):
            current_node = path[i]
            next_node = path[i + 1]
            distance += self.distance_matrix[current_node][next_node]

        return distance

    def run(self, num_iterations):
        best_path = None
        best_distance = np.inf

        for iteration in range(num_iterations):
            ant_paths = self.generate_ant_paths()
            self.update_pheromone(ant_paths)

            # Update best path
            if ant_paths[0][1] < best_distance:
                best_distance = ant_paths[0][1]
                best_path = ant_paths[0][0]

            # Evaporate pheromone
            self.pheromone_matrix *= self.evaporation_rate

        return best_path, best_distance

    def generate_ant_paths(self):
        ant_paths = []

        for ant in range(self.num_ants):
            visited = [False] * self.num_nodes
            path = []
            current_node = np.random.randint(self.num_nodes)
            visited[current_node] = True
            path.append(current_node)

            for _ in range(self.num_nodes - 1):
                next_node = self.choose_next_node(current_node, visited)  # Usuń argument distance_matrix
                visited[next_node] = True
                path.append(next_node)
                current_node = next_node

            path.append(path[0])  # Complete the cycle
            path_distance = self.calculate_path_distance(path)  # Usuń argumenty pheromone_matrix i distance_matrix
            ant_paths.append((path, path_distance))

        ant_paths.sort(key=lambda x: x[1])  # Sort paths by distance
        return ant_paths

    def choose_next_node(self, current_node, visited):
        unvisited_nodes = np.where(np.logical_not(visited))[0]
        pheromone_values = self.pheromone_matrix[current_node, unvisited_nodes]
        attractiveness_values = 1.0 / (self.calculate_path_distance(unvisited_nodes) + 1e-6)  # Usuń argumenty current_node i unvisited_nodes
        probabilities = pheromone_values ** self.alpha * attractiveness_values ** self.beta
        probabilities /= probabilities.sum()

        next_node = np.random.choice(unvisited_nodes, p=probabilities)
        return next_node

    def update_pheromone(self, ant_paths):
        for path, distance in ant_paths:
            for i in range(self.num_nodes):
                current_node = path[i]
                next_node = path[i+1]
                self.pheromone_matrix[current_node, next_node] += self.pheromone_deposit / distance

def animate(i, ant_paths_all):
    ax.clear()
    ax.set_title(f'Iteracja {i+1} - Funkcja sferyczna')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wartość funkcji')

    ant_paths = ant_paths_all[i]
    for ant_path in ant_paths:
        positions = np.array(ant_paths)  # Zmiana na tablicę numpy
        scores = [path[1] for path in ant_paths]  # Zmiana na odczytanie wartości funkcji z odpowiedniego indeksu
        best_position = positions[np.argmax(scores)]
        best_score = np.max(scores)

        ax.scatter(positions[:, 0], positions[:, 1], [sphere_function(position[0], position[1]) for position in positions], color='b', s=40, alpha=1.0)
        ax.scatter(best_position[0], best_position[1], sphere_function(best_position[0], best_position[1]), color='r', marker='*', s=200, label='Najlepsza pozycja')

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = sphere_function(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    ax.legend()

# Tworzenie macierzy odległości
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
distance_matrix = sphere_function(X, Y)

# Parametry algorytmu mrówkowego
num_ants = 10
num_iterations = 100

# Inicjalizacja algorytmu mrówkowego
ant_colony = AntColony(num_ants=num_ants, num_nodes=100, distance_matrix=distance_matrix, alpha=1, beta=2, evaporation_rate=0.5, pheromone_deposit=1)

# Inicjalizacja mrówek
ant_paths_all = []  # Lista przechowująca ścieżki mrówek dla każdej iteracji
for _ in range(num_iterations):
    ant_paths = ant_colony.generate_ant_paths()
    ant_paths_all.append([path[0] for path in ant_paths])  # Zmieniona zapis pozycji mrówek

# Inicjalizacja wykresu dla funkcji sferycznej
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Utworzenie animacji
animation = FuncAnimation(fig, animate, frames=num_iterations, interval=200, fargs=(ant_paths_all,))
plt.show()
