import numpy as np
import matplotlib.pyplot as plt

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

class Ant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = sphere_function(x, y)

def update_ant(ant, step_size):
    ant.x += np.random.uniform(-step_size, step_size)
    ant.y += np.random.uniform(-step_size, step_size)
    ant.z = sphere_function(ant.x, ant.y)

def ant_algorithm(num_ants, num_dimensions, num_steps, step_size):
    ants = [Ant(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(num_ants)]
    fitness_values = []

    for _ in range(num_steps):
        for ant in ants:
            update_ant(ant, step_size)
            fitness_values.append(ant.z)

    return fitness_values

# Parametry algorytmu
num_ants = 100
num_dimensions = 2
num_steps = 100
step_size = 0.1

# Wywołanie algorytmu mrówkowego
fitness_values = ant_algorithm(num_ants, num_dimensions, num_steps, step_size)

# Wygenerowanie wykresu
plt.plot(range(1, len(fitness_values) + 1), fitness_values)
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji celu')
plt.title('Wykres wartości funkcji celu dla każdej cząsteczki')
plt.show()
