import sys
import numpy as np
from tqdm import tqdm
import os

path = sys.argv[1]
data_type = sys.argv[2] # int/float

# data = np.loadtxt(path, delimiter=',', dtype='int')
# outpath = ".".join(path.split('.')[:-1]) + '.npy'
# np.save(outpath, data)

print(f"Save {path.split('/')[3]} as .np ...", end = '')
def save_to_npy(input_path):
    # read
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # longest row
    max_len = max(len(line.strip().split(',')) for line in lines)
    
    # rows have different lengths: pad with -1 (must be dropped when used)
    n_days = len(lines)
    if data_type == 'int':
        processed_data = np.full((n_days, max_len), -1, dtype=np.int32)
        for i, line in enumerate(lines):
            tags = list(map(int, line.strip().split(',')))
            processed_data[i, :len(tags)] = tags
    elif data_type == 'float':
        processed_data = np.full((n_days, max_len), -1.0, dtype=np.float32)
        for i, line in enumerate(lines):
            tags = list(map(float, line.strip().split(',')))
            processed_data[i, :len(tags)] = tags
    # save as .npy and delete the text original
    outpath = ".".join(input_path.split('.')[:-1]) + '.npy'
    np.save(outpath, processed_data)
    os.remove(input_path)
save_to_npy(path)

print("OK")