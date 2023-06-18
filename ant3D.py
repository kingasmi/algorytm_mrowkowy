#Funkcja celu - sfera - 3D 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def sphere_function(x, y):
    """Funkcja celu - sfera"""
    return x**2 + y**2

class Ant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = sphere_function(x, y)

def update_ant(ant, step_size):
    new_x = ant.x + np.random.uniform(-step_size, step_size)
    new_y = ant.y + np.random.uniform(-step_size, step_size)
    new_z = sphere_function(new_x, new_y)
    
    if new_z < ant.z:
        ant.x = new_x
        ant.y = new_y
        ant.z = new_z

def plot_animation(ants, num_steps, step_size):
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    fig_2d = plt.figure()
    ax_2d = fig_2d.add_subplot(111)
    
    x_range = np.linspace(-10, 10, 200)  # Zwiększenie liczby próbek x
    y_range = np.linspace(-10, 10, 200)  # Zwiększenie liczby próbek y
    X, Y = np.meshgrid(x_range, y_range)
    Z = sphere_function(X, Y)
    
    ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ant_points = ax_3d.scatter([ant.x for ant in ants], [ant.y for ant in ants], [ant.z for ant in ants], color='blue')
    best_ant = min(ants, key=lambda ant: ant.z)
    best_point = ax_3d.scatter(best_ant.x, best_ant.y, best_ant.z, color='red')
    iter_text_3d = ax_3d.text2D(0.05, 0.95, "", transform=ax_3d.transAxes)
    best_text_3d = ax_3d.text2D(0.05, 0.90, "", transform=ax_3d.transAxes)
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('Algorytm mrówkowy - funkcja sfera')
    
    iter_text_2d = ax_2d.text(0.05, 0.95, "", transform=ax_2d.transAxes)
    best_text_2d = ax_2d.text(0.05, 0.90, "", transform=ax_2d.transAxes)
    
    def update(frame):
        for ant in ants:
            update_ant(ant, step_size)
        
        ant_points._offsets3d = ([ant.x for ant in ants], [ant.y for ant in ants], [ant.z for ant in ants])
        best_ant = min(ants, key=lambda ant: ant.z)
        best_point._offsets3d = ([best_ant.x], [best_ant.y], [best_ant.z])
        iter_text_3d.set_text(f"Iteracja: {frame+1}")
        best_text_3d.set_text(f"Najlepszy wynik: {best_ant.z:.7f}")
        
        ax_2d.clear()
        ax_2d.contourf(X, Y, Z, cmap='viridis', alpha=0.8)
        ax_2d.plot([ant.x for ant in ants], [ant.y for ant in ants], 'bo')
        ax_2d.plot(best_ant.x, best_ant.y, 'ro')
        iter_text_2d.set_text(f"Iteracja: {frame+1}")
        best_text_2d.set_text(f"Najlepszy wynik: {best_ant.z:.7f}")
        
    anim = animation.FuncAnimation(fig_3d, update, frames=num_steps, interval=200, repeat=False)
    anim_2d = animation.FuncAnimation(fig_2d, update, frames=num_steps, interval=200, repeat=False)
    
    # Zapisywanie animacji do plików
    anim.save('sphere_3d.gif', writer='imagemagick')
    anim_2d.save('sphere_2d.gif', writer='imagemagick')
    
    plt.show()

# Tworzenie populacji mrówek
num_ants = 100
ants = [Ant(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(num_ants)]

# Uruchomienie animacji
num_steps = 100
step_size = 0.1
plot_animation(ants, num_steps, step_size)
