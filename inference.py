import taichi as ti
import numpy as np
import yaml
import os
from src.model import DiffMPMModel
from src.dataset import Scene, robot, fish
from src.utils import visualize

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
    
    # Load weights
    if os.path.exists('checkpoints/weights.npy'):
        model.weights.from_numpy(np.load('checkpoints/weights.npy'))
        model.bias.from_numpy(np.load('checkpoints/bias.npy'))
        print("Loaded weights from checkpoints/")
    else:
        print("No weights found, using random initialization.")
            
    # Initialize particles
    for i in range(scene.n_particles):
        model.x[0, i] = scene.x[i]
        model.F[0, i] = [[1, 0], [0, 1]]
        model.actuator_id[i] = scene.actuator_id[i]
        model.particle_type[i] = scene.particle_type[i]

    gui = ti.GUI("Differentiable MPM Inference", (640, 640), background_color=0xFFFFFF, show_gui=False)

    print("Running inference...")
    total_steps = 1500 # Longer simulation for inference
    for s in range(total_steps - 1):
        model.compute_actuation(s)
        model.clear_grid()
        model.p2g(s)
        model.grid_op(3, 0.5)
        model.g2p(s)
        
        if s % 16 == 0:
            visualize(model, s, 'results/inference_frames', gui)
            
    print(f"Inference complete. Frames saved to results/inference_frames/")

if __name__ == '__main__':
    main()
