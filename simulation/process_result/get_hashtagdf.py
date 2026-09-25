import sys
import numpy as np
import pandas as pd  # type: ignore

path = sys.argv[1]
path = ".".join(path.split('.')[:-1]) + '.npy'

N_DAYS = int(sys.argv[2])
N_SAVE_FROM_LAST_STEP = int(sys.argv[3])

print("Get hashtag time seires dataframe...", end = '')
post_record = np.load(path)
flattened_data = post_record.flatten()
all_tag = pd.Series(pd.Series(flattened_data[flattened_data != -1]).unique(), dtype=np.int32)

daily_counts = {}
for i in range(post_record.shape[0]):
    daily_counts[i] = pd.Series(post_record[i], dtype=np.int32).value_counts()
word_fulltime_df = pd.DataFrame(daily_counts, index=all_tag)
word_fulltime_df.columns = range(N_DAYS - N_SAVE_FROM_LAST_STEP, N_DAYS, 1)  # column = model day

outpath = ".".join(path.split('.')[:-1]) + '.pkl'
word_fulltime_df.iloc[:-1, :].to_pickle(outpath)
print('OK')
