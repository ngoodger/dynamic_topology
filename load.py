import numpy as np
from load_topology_dataset import IMAGE_SIZE

def calculate_cell_load(cells, loads):
    if len(cells) == 0:
        return None
    elif len(loads):
        
        cells_np = (np.array(cells) / IMAGE_SIZE).reshape((len(cells), 2))
        loads_np = (np.array(loads) / IMAGE_SIZE).reshape((len(loads), 2))
        #print(loads_np.shape)
        #.reshape((len(loads), 2))
        load_cells_np = np.zeros([len(cells)])
        #print(cells_np)
        #for i, load_loc in enumerate(loads_np):
        for i, load_loc in enumerate(loads_np):
            diff_loc = cells_np - load_loc
            euclidean_dist = np.sqrt(np.sum(np.square(diff_loc), axis=1))
            # Traffic is attributed to cell by square distance.
            load_cell = square_dist(euclidean_dist)

            load_cells_np = load_cells_np + load_cell
        load_cells = []
        # Sum of loads should always equal number of loads
        #print(load_cells_np.sum())
        assert(np.isclose(load_cells_np.sum(), len(loads)))

        for load, cell in zip(load_cells_np, cells):
            x, y = cell[0], cell[1]
            load_cells.append((x, y, load))
        return load_cells

def linear_dist(x):
    # Subtract from 2 since euclidean distance can be more than 1.
    load = 2. - 10*x
    return (load / load.sum())   

def square_dist(x):
    load = 2. - np.square(x)
    return (load / load.sum())
