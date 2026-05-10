import taichi as ti
import numpy as np
import yaml
import os
import csv
from src.model import DiffMPMModel
from src.dataset import Scene, robot, fish
from src.utils import plot_losses

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('config.yaml')
    
    ti.init(arch=ti.gpu)
    
    model = DiffMPMModel(
        dim=config['dim'], 
        n_grid=config['n_grid'], 
        max_steps=config['max_steps'],
        dt=config['dt']
    )
    
    scene = Scene(model.dx)
    if config['scene_type'] == 'robot':
        n_actuators = robot(scene)
    else:
        n_actuators = fish(scene)
        
    model.allocate_fields(scene.n_particles, n_actuators)
    model.n_solid_particles = scene.n_solid_particles
    
    # Initialize weights
    for i in range(model.n_actuators):
        for j in range(model.n_sin_waves):
            model.weights[i, j] = np.random.randn() * 0.01
            
    # Initialize particles
    for i in range(scene.n_particles):
        model.x[0, i] = scene.x[i]
        model.F[0, i] = [[1, 0], [0, 1]]
        model.actuator_id[i] = scene.actuator_id[i]
        model.particle_type[i] = scene.particle_type[i]

    def forward(total_steps=config['steps']):
        for s in range(total_steps - 1):
            model.compute_actuation(s)
            model.clear_grid()
            model.p2g(s)
            model.grid_op(3, 0.5) # bound=3, coeff=0.5
            model.g2p(s)
        
        model.x_avg[None] = [0, 0]
        model.compute_x_avg(total_steps)
        model.compute_loss()

    losses = []
    os.makedirs('results', exist_ok=True)
    log_file = open('results/training_log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['iteration', 'loss'])

    print("Starting training...")
    for iter in range(config['iters']):
        with ti.ad.Tape(model.loss):
            forward()
            
        l = model.loss[None]
        losses.append(l)
        print(f'i={iter} loss={l}')
        log_writer.writerow([iter, l])
        
        # Gradient descent
        for i in range(model.n_actuators):
            for j in range(model.n_sin_waves):
                model.weights[i, j] -= config['learning_rate'] * model.weights.grad[i, j]
            model.bias[i] -= config['learning_rate'] * model.bias.grad[i]

    log_file.close()
    
    # Save results
    plot_losses(losses, save_path='results/training_curve.png')
    
    # Save weights (as a simple numpy file for now)
    os.makedirs('checkpoints', exist_ok=True)
    np.save('checkpoints/weights.npy', model.weights.to_numpy())
    np.save('checkpoints/bias.npy', model.bias.to_numpy())
    print("Training complete. Weights saved to checkpoints/")

if __name__ == '__main__':
    main()
