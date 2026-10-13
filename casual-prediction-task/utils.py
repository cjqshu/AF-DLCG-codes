import numpy as np
import pandas as pd

def array_to_batch(data, batch_size):
    num_batches = np.floor(len(data) / batch_size)
    if len(data) % batch_size == 0:
        batches = np.array_split(data, num_batches)
    else:
        batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)
    return np.array(batches)


def normalization(data):
    data_normalize = data.copy()
    cols = data_normalize.columns
    i = data.shape[-1] - 2 
    data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - data.loc[:, cols[i]].min()) / (
            data.loc[:, cols[i]].max() - data.loc[:, cols[i]].min())
    return data_normalize


def normalize(data):
    import numpy as np
    data_normalize = data.copy()
    cols = data_normalize.columns
    for i in range(data.shape[-1]):
        col_mean = data.loc[:, cols[i]].mean()
        col_std = data.loc[:, cols[i]].std()
        
        if col_std == 0 or np.isnan(col_std) or np.isinf(col_std):
            data_normalize.loc[:, cols[i]] = 0.0
        else:
            data_normalize.loc[:, cols[i]] = (data.loc[:, cols[i]] - col_mean) / col_std
    return data_normalize


def inverse_normalize(test_df, pred_y):
    test_y_mean = test_df.loc[:, "CPT_Y"].mean()  
    test_y_std = test_df.loc[:, "CPT_Y"].std()
    inverse_normalize_pred_y = pred_y * test_y_std + test_y_mean
    return inverse_normalize_pred_y
