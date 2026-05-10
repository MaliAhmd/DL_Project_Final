import taichi as ti
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize(model, s, folder, gui):
    aid = model.actuator_id.to_numpy()
    colors = np.empty(shape=model.n_particles, dtype=np.uint32)
    particles = model.x.to_numpy()[s]
    actuation_ = model.actuation.to_numpy()
    for i in range(model.n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def plot_losses(losses, save_path=None):
    plt.title("Optimization Loss")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    if save_path:
        plt.savefig(save_path)
    plt.show()
