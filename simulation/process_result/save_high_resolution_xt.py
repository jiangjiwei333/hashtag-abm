import sys
import pandas as pd

x_path = sys.argv[1]
N_DAYS = int(sys.argv[2])
N_SAVE_FROM_LAST_STEP = int(sys.argv[3])

stationary_day = N_DAYS - N_SAVE_FROM_LAST_STEP + 1
start_line = 0
begin_t = 24 * 6 * stationary_day
x_df = []
with open(x_path, 'r') as f_x:
    # skip the days that were not saved
    for _ in range(start_line):
        next(f_x)

    i = begin_t
    for x_line in f_x:
        x_tmp = pd.DataFrame({
            'time': i,
            'tag': map(int, x_line.strip().split(',')[0::2]),
            'value': map(int, x_line.strip().split(',')[1::2])})
        x_df.append(x_tmp)
        i += 1
x_df = pd.concat(x_df, axis=0)
x_df.to_pickle(".".join(x_path.split('.')[:-1]) + '.pkl')

