import numpy as np
import torch
from torch.utils.data import Dataset
import random
import load

IMAGE_SIZE = 32
MAX_CELL_COUNT = 8
MIN_CELL_COUNT = 2
OVERRIDE_TEST = True

class LoadCellDataset(Dataset):
    
    def __init__(self, initial_cell_counts, initial_load_counts,
                 input_seq_len, target_seq_len, network_mutate_prob=[0.5, 0.75], seed=None):

        self.initial_cell_counts = initial_cell_counts
        self.initial_load_counts = initial_load_counts
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.network_mutate_prob = network_mutate_prob    
        if seed:
            random.seed(seed)
        
    def _add_random_cell(self, cells, cells_grid):
        free_cell_locations = np.where(cells_grid==0)
        x = np.random.choice(free_cell_locations[0])
        y = np.random.choice(free_cell_locations[1])
        cells_grid[x, y] = 1
        cells.append((x,y))
        return cells, cells_grid
        
    def _remove_random_cell(self, cells, cells_grid, reference_cell=None, remove_reference_cell=True):
        cell_idx = random.randrange(len(cells))
        if remove_reference_cell or not (cells[cell_idx][0] == reference_cell[0] and cells[cell_idx][1] == reference_cell[1]):
            x, y = cells.pop(random.randrange(len(cells)))
            cells_grid[x, y] = 0
        return cells, cells_grid
    
    def _add_random_load(self, loads, loads_grid):
        free_load_locations = np.where(loads_grid==0)
        x = np.random.choice(free_load_locations[0])
        y = np.random.choice(free_load_locations[1])
        loads_grid[x, y] = 1
        loads.append((x, y))
        return loads, loads_grid
        
    def _remove_random_load(self, cells, loads_grid):
        x, y = cells.pop(random.randrange(len(loads)))
        loads_grid[x, y] = 0
        return loads, loads_grid
    
    def __len__(self):
        return 1000000000

    @staticmethod
    def build_inputs_and_targets(input_seq_len, target_seq_len, ref_x, ref_y, load_cells_seq_input, load_cells_seq_target):
        reference_cell_input = np.zeros((input_seq_len, 1))
        reference_cell_present_input = np.zeros((input_seq_len, 1))
        neighbourhood_cell_rel_input = np.zeros((input_seq_len, MAX_CELL_COUNT, 2))
        neighbourhood_cell_load_input = np.zeros((input_seq_len, MAX_CELL_COUNT, 1))
        neighbourhood_cell_present_input = np.zeros((input_seq_len, MAX_CELL_COUNT, 1))
        for seq_idx in range(input_seq_len):
            reference_cell_active = False
            cell_idx = 0
            for cell in load_cells_seq_input[seq_idx]:
                current_x = cell[0]
                current_y = cell[1]
                current_load = cell[2]
                # If not reference cell add to neighbourhood
                if not (current_x == ref_x and current_y == ref_y):
                    # neighbourhood normalized by dividing by IMAGE_SIZE.
                    # Subtract from 1 since cells that are closes have greated influence.
                    neighbourhood_cell_rel_input[seq_idx, cell_idx, 0] = 1. - abs(ref_x - current_x) / IMAGE_SIZE
                    neighbourhood_cell_rel_input[seq_idx, cell_idx, 1] = 1. - abs(ref_y - current_y) / IMAGE_SIZE
                    neighbourhood_cell_load_input[seq_idx, cell_idx] = current_load
                    neighbourhood_cell_present_input[seq_idx, cell_idx] = 1.
                    cell_idx += 1
                else:
                    reference_cell_input[seq_idx] = current_load
                    reference_cell_active = True
            
            if reference_cell_active:
                reference_cell_present_input[seq_idx] = 1.0
            
        reference_cell_target = np.zeros((target_seq_len, 1))
        reference_cell_present_target = np.zeros((target_seq_len, 1))
        neighbourhood_cell_rel_target = np.zeros((target_seq_len, MAX_CELL_COUNT, 2))
        neighbourhood_cell_present_target = np.zeros((target_seq_len, MAX_CELL_COUNT, 1))
        for seq_idx in range(target_seq_len):
            reference_cell_active = False
            cell_idx = 0
            for cell in load_cells_seq_target[seq_idx]:
                current_x = cell[0]
                current_y = cell[1]
                current_load = cell[2]
                # If not reference cell add to neighbourhood
                if not (current_x == ref_x and current_y == ref_y):
                    # neighbourhood normalized by dividing by IMAGE_SIZE.
                    # Subtract from 1 since cells that are closes have greated influence.
                    neighbourhood_cell_rel_target[seq_idx, cell_idx, 0] = 1. - abs(ref_x - current_x) / IMAGE_SIZE
                    neighbourhood_cell_rel_target[seq_idx, cell_idx, 1] = 1. - abs(ref_y - current_y) / IMAGE_SIZE
                    neighbourhood_cell_present_target[seq_idx, cell_idx] = 1.
                    cell_idx += 1
                else:
                    reference_cell_target[seq_idx] = current_load
                    reference_cell_active = True
            if reference_cell_active:
                reference_cell_present_target[seq_idx] = 1.0
        return (torch.tensor(reference_cell_input, dtype=torch.float32),
                torch.tensor(reference_cell_present_input, dtype=torch.float32),
                torch.tensor(neighbourhood_cell_rel_input, dtype=torch.float32), 
                torch.tensor(neighbourhood_cell_load_input, dtype=torch.float32),
                torch.tensor(neighbourhood_cell_present_input, dtype=torch.float32),
                torch.tensor(reference_cell_target, dtype=torch.float32),
                torch.tensor(reference_cell_present_target, dtype=torch.float32),
                torch.tensor(neighbourhood_cell_rel_target, dtype=torch.float32),
                torch.tensor(neighbourhood_cell_present_target, dtype=torch.float32),)
        
    def __getitem__(self, idx):
        cells, loads = [], []
        cells_grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        loads_grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)
        initial_cell_count = random.randint(self.initial_cell_counts[0], self.initial_cell_counts[1])
        initial_load_count = random.randint(self.initial_load_counts[0], self.initial_load_counts[1])
        for i in range(initial_cell_count):
            cells, cells_grid = self._add_random_cell(cells, cells_grid)
        for i in range(initial_load_count):
            loads, loads_grid = self._add_random_load(loads, loads_grid)
        
        
        ################################################################
        # Generate input sequence
        ################################################################
        load_cells_seq_input = []
        # Generate input cells
        for i in range(self.input_seq_len):
            load_cells = load.calculate_cell_load(cells, loads)
            load_cells_seq_input.append(load_cells)
            for cell_idx in range(MAX_CELL_COUNT):
                mutate_roll = random.random()
                # Only add cells if cell count is less than MAX_CELL_COUNT otherwise do nothing.
                if mutate_roll < self.network_mutate_prob[0] and mutate_roll < self.network_mutate_prob[1] and (len(cells) < MAX_CELL_COUNT):
                    cells, cells_grid = self._add_random_cell(cells, cells_grid)
                if mutate_roll < self.network_mutate_prob[1] and (len(cells) >= MIN_CELL_COUNT):
                    cells, cells_grid = self._remove_random_cell(cells, cells_grid)
                
        ################################################################
        # Get a random reference cell present in the last element of the input sequence.
        # The last element is chosen to allow for examples where neighbourhood exists before reference cell.
        ################################################################

        reference_cell = load_cells_seq_input[-1][random.randrange(len(load_cells_seq_input[-1]))]
        ref_x, ref_y = reference_cell[0], reference_cell[1]
        
        ################################################################
        # Generate target sequence
        ################################################################
        load_cells_seq_target = []
        for i in range(self.target_seq_len):
            load_cells = load.calculate_cell_load(cells, loads)
            load_cells_seq_target.append(load_cells)
            for cell_idx in range(MAX_CELL_COUNT):
                mutate_roll = random.random()
                if mutate_roll < self.network_mutate_prob[0] and mutate_roll < self.network_mutate_prob[1]  and (len(cells) < MAX_CELL_COUNT):
                    cells, cells_grid = self._add_random_cell(cells, cells_grid,)
                elif mutate_roll < self.network_mutate_prob[1] and (len(cells) >= MIN_CELL_COUNT):
                    cells, cells_grid = self._remove_random_cell(cells, cells_grid, reference_cell, remove_reference_cell=False)
                                          
        # Now we know the reference cell.  Build the inputs
        (reference_cell_input,
        reference_cell_present_input,
        neighbourhood_cell_rel_input, 
        neighbourhood_cell_load_input,
        neighbourhood_cell_present_input,
        reference_cell_target,
        reference_cell_present_target,
        neighbourhood_cell_rel_target,
        neighbourhood_cell_present_target,) = self.build_inputs_and_targets(self.input_seq_len, self.target_seq_len, ref_x, ref_y, load_cells_seq_input, load_cells_seq_target)

        return (reference_cell_input,
                reference_cell_present_input,
                neighbourhood_cell_rel_input, 
                neighbourhood_cell_load_input,
                neighbourhood_cell_present_input,
                reference_cell_target,
                reference_cell_present_target,
                neighbourhood_cell_rel_target,
                neighbourhood_cell_present_target,)
