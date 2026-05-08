"""
Evolutionary Algorithm

Runs a Genetic Algorithm to optimize bounding box 
placement around LiDAR point clusters. It generates random bounding boxes and 
incrementally mutates them, keeping mutations that improve the 'fitness' (how well 
the box covers the points while minimizing empty space). Used to benchmark 
the hardware latency of a genetic algorithm for the problem.

"""

import numpy as np
import os
import glob
import time
import argparse

class GeneticAlg:
    def __init__(self, near_field=10.0, grid_size=32):
        self.near_field = near_field
        self.grid_size = grid_size
    
    def get_bounds(self):
        return [(0, self.near_field), (-self.near_field/2, self.near_field/2), 
                (-2, 2), (0.5, 6.0), (0.5, 6.0), (0.5, 3.0), (0, 2*np.pi)]
    
    def get_occupied_coords(self, points):
        mask = ((points[:, 0] > 0) & (points[:, 0] < self.near_field) &
                (points[:, 1] > -self.near_field/2) & (points[:, 1] < self.near_field/2) &
                (points[:, 2] > -2) & (points[:, 2] < 2))
        pts = points[mask]
        
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        if len(pts) > 0:
            x_norm = (pts[:, 0] / self.near_field) * (self.grid_size - 1)
            y_norm = ((pts[:, 1] + self.near_field/2) / self.near_field) * (self.grid_size - 1)
            z_norm = ((pts[:, 2] + 2) / 4) * (self.grid_size - 1)
            grid[np.clip(x_norm.astype(int), 0, self.grid_size - 1),
                 np.clip(y_norm.astype(int), 0, self.grid_size - 1),
                 np.clip(z_norm.astype(int), 0, self.grid_size - 1)] = 1
                 
        indices = np.argwhere(grid == 1)
        if len(indices) == 0: return np.empty((0, 3))
        
        x_world = (indices[:, 0] / (self.grid_size - 1)) * self.near_field
        y_world = -self.near_field/2 + (indices[:, 1] / (self.grid_size - 1)) * self.near_field
        z_world = -2 + (indices[:, 2] / (self.grid_size - 1)) * 4
        return np.column_stack((x_world, y_world, z_world))
    
    def fitness(self, occupied_points, bbox):
        cx, cy, cz, sx, sy, sz, _ = bbox
        if len(occupied_points) == 0: return 0.0
            
        dx = np.abs(occupied_points[:, 0] - cx)
        dy = np.abs(occupied_points[:, 1] - cy)
        dz = np.abs(occupied_points[:, 2] - cz)
        
        in_box = (dx < sx/2) & (dy < sy/2) & (dz < sz/2)
        coverage = np.sum(in_box) / len(occupied_points)
        size_penalty = (sx * sy * sz) / 1000.0
        
        return coverage - 0.05 * size_penalty

    def optimize(self, points, pop_size=30, gens=50):
        start_time = time.time()
        occupied = self.get_occupied_coords(points)
        bounds = self.get_bounds()
        
        pop = [[np.random.uniform(b[0], b[1]) for b in bounds] for _ in range(pop_size)]
        scores = [self.fitness(occupied, ind) for ind in pop]
        
        for _ in range(gens):
            for i in range(pop_size):
                mutant = [pop[i][j] + np.random.normal(0, 0.5) for j in range(7)]
                mutant = [np.clip(mutant[j], bounds[j][0], bounds[j][1]) for j in range(7)]
                
                mutant_score = self.fitness(occupied, mutant)
                if mutant_score >= scores[i]:
                    pop[i] = mutant
                    scores[i] = mutant_score
                    
        best_idx = np.argmax(scores)
        return pop[best_idx], scores[best_idx], (time.time() - start_time) * 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GENETIC ALGORITHM BENCHMARK")
    print("="*60)
    
    optimizer = GeneticAlg()
    bin_files = sorted(glob.glob("dataset/training/velodyne/*.bin"))
    
    results = []
    for bin_file in bin_files[:args.num_samples]:
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        if len(points) > 0:
            _, score, opt_time = optimizer.optimize(points)
            results.append({'fitness': score, 'time': opt_time})
            
    if results:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Evaluated Samples: {len(results)}")
        print(f"Average Fitness:   {np.mean([r['fitness'] for r in results]):.4f}")
        print(f"Average Latency:   {np.mean([r['time'] for r in results]):.2f} ms")

if __name__ == "__main__":
    main()