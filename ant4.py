import numpy as np
import matplotlib.pyplot as plt

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

def rastrigin_function(x, y):
    """Funkcja celu - Rastrigin"""
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)

# Utworzenie siatki punktów
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Obliczenie wartości funkcji dla każdego punktu na siatce
Z_sphere = sphere_function(X, Y)
Z_rastrigin = rastrigin_function(X, Y)

# Wyświetlanie wykresu dla funkcji sfera
fig_sphere = plt.figure()
ax_sphere = fig_sphere.add_subplot(111, projection='3d')
ax_sphere.plot_surface(X, Y, Z_sphere, cmap='viridis')
ax_sphere.set_xlabel('X')
ax_sphere.set_ylabel('Y')
ax_sphere.set_zlabel('Z')
ax_sphere.set_title('Funkcja sfera')
plt.savefig('Funkcja_sfera.png')

# Wyświetlanie wykresu dla funkcji Rastrigina
fig_rastrigin = plt.figure()
ax_rastrigin = fig_rastrigin.add_subplot(111, projection='3d')
ax_rastrigin.plot_surface(X, Y, Z_rastrigin, cmap='viridis')
ax_rastrigin.set_xlabel('X')
ax_rastrigin.set_ylabel('Y')
ax_rastrigin.set_zlabel('Z')
ax_rastrigin.set_title('Funkcja Rastrigina')
plt.savefig('Funkcja_Rastrigina.png')

plt.show()
