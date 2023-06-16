import numpy as np
import matplotlib.pyplot as plt

# Definicja funkcji celu
def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

def rastrigin_function(x, y):
    """Funkcja celu - Rastrigin"""
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

# Implementacja algorytmu mrówkowego
def ant_colony_optimization(objective_function, num_ants, num_iterations, bounds):
    # Inicjalizacja parametrów
    alpha = 1.0  # Wpływ feromonów
    beta = 1.0  # Wpływ heurystyki
    rho = 0.5   # Współczynnik parowania feromonów
    Q = 100     # Ilość feromonów pozostawianych przez mrówkę
    num_dimensions = 2  # Liczba wymiarów (x, y)

    # Inicjalizacja feromonów na krawędziach
    tau = np.ones((num_dimensions, num_dimensions))

    # Inicjalizacja najlepszego rozwiązania
    best_solution = None
    best_fitness = float('inf')

    # Inicjalizacja tablicy przechowującej wartości funkcji fitness w kolejnych iteracjach
    fitness_history = []

    # Inicjalizacja tablicy przechowującej średnie wartości funkcji celu dla każdej cząsteczki
    mean_fitness_per_particle = np.zeros((num_iterations, num_ants))

    # Główna pętla algorytmu
    for iteration in range(num_iterations):
        solutions = np.zeros((num_ants, num_dimensions))
        fitness_values = np.zeros(num_ants)
        
        # Inicjalizacja tablicy feromonowej
        delta_tau = np.zeros((num_dimensions, num_dimensions))

        # Poruszanie się mrówek i obliczanie wartości funkcji fitness
        for ant in range(num_ants):
            current_solution = solutions[ant]
            for i in range(num_dimensions):
                # Obliczanie atrakcyjności przejść
                attractiveness = (tau[i] ** alpha) * (1.0 / (objective_function(*current_solution) + 1e-10)) ** beta
                probabilities = attractiveness / np.sum(attractiveness)

                # Wybór kolejnego wierzchołka
                next_vertex = np.random.choice(range(num_dimensions), p=probabilities)
                current_solution[i] = bounds[next_vertex, 0] + (bounds[next_vertex, 1] - bounds[next_vertex, 0]) * np.random.rand()

            solutions[ant] = current_solution
            fitness_values[ant] = objective_function(*current_solution)

            # Aktualizacja najlepszego rozwiązania
            if fitness_values[ant] < best_fitness:
                best_solution = current_solution
                best_fitness = fitness_values[ant]

            # Aktualizacja feromonów
            for ant in range(num_ants):
                for i in range(num_dimensions - 1):
                    delta_tau[int(solutions[ant, i]), int(solutions[ant, i+1])] += Q / fitness_values[ant]
                if num_dimensions == 2:
                    # Dla 2 wymiarów dodajemy feromony tylko dla ostatniego -> pierwszego wierzchołka
                    delta_tau[int(solutions[ant, -1]), int(solutions[ant, 0])] += Q / fitness_values[ant]

            tau = (1 - rho) * tau + delta_tau


        # Zapisanie wartości najlepszej funkcji fitness w kolejnych iteracjach
        fitness_history.append(best_fitness)

        # Obliczanie średniej wartości funkcji celu dla każdej cząsteczki
        mean_fitness_per_particle[iteration] = np.mean(fitness_values)

    return best_solution, best_fitness, fitness_history, mean_fitness_per_particle

# Parametry algorytmu
num_ants = 10
num_iterations = 100
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])  # Zakres dla x i y

# Wywołanie algorytmu dla funkcji celu - sfera
fitness_history_sphere, mean_fitness_per_particle_sphere = ant_colony_optimization(sphere_function, num_ants, num_iterations, bounds)

# Wywołanie algorytmu dla funkcji celu - Rastrigin
fitness_history_rastrigin, mean_fitness_per_particle_rastrigin = ant_colony_optimization(rastrigin_function, num_ants, num_iterations, bounds)

# Wykres zmiany wartości funkcji fitness dla funkcji sferycznej
plt.figure()
x = np.arange(num_iterations)
for ant in range(num_ants):
    plt.plot(x, fitness_history_sphere[:, ant], label=f'Cząsteczka {ant+1}')
plt.xlabel('Iteracje')
plt.ylabel('Wartość funkcji fitness')
plt.legend()
plt.title('Zmiana wartości funkcji fitness dla funkcji sferycznej')

# Wykres zmiany wartości funkcji fitness dla funkcji Rastrigina
plt.figure()
x = np.arange(num_iterations)
for ant in range(num_ants):
    plt.plot(x, fitness_history_rastrigin[:, ant], label=f'Cząsteczka {ant+1}')
plt.xlabel('Iteracje')
plt.ylabel('Wartość funkcji fitness')
plt.legend()
plt.title('Zmiana wartości funkcji fitness dla funkcji Rastrigina')

# Wykres średniej wartości funkcji celu dla każdej cząsteczki dla funkcji sferycznej
plt.figure()
x = np.arange(num_iterations)
for ant in range(num_ants):
    plt.plot(x, mean_fitness_per_particle_sphere[:, ant], label=f'Cząsteczka {ant+1}')
plt.xlabel('Iteracje')
plt.ylabel('Średnia wartość funkcji celu')
plt.legend()
plt.title('Średnia wartość funkcji celu dla każdej cząsteczki - funkcja sferyczna')

# Wykres średniej wartości funkcji celu dla każdej cząsteczki dla funkcji Rastrigina
plt.figure()
x = np.arange(num_iterations)
for ant in range(num_ants):
    plt.plot(x, mean_fitness_per_particle_rastrigin[:, ant], label=f'Cząsteczka {ant+1}')
plt.xlabel('Iteracje')
plt.ylabel('Średnia wartość funkcji celu')
plt.legend()
plt.title('Średnia wartość funkcji celu dla każdej cząsteczki - funkcja Rastrigina')

# Wyświetlenie wszystkich wykresów
plt.show()
