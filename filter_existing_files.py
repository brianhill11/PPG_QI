import os
import numpy as np
from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
import pickle
import glob
from tqdm import tqdm
import sys

sys.path.append("../ABP_pred")
import train_ppg_qi
from src.project_configs import project_dir, train_dir, val_dir, test_dir, window_size, ppg_col


ppg_qi_model_weights = os.path.join("weights", "ppg_qi_model.hdf5")
ppg_qi_scaler_file = os.path.join("ppg_qi_scaler.pkl")

ppg_qi_model = train_ppg_qi.create_model()
ppg_qi_model.load_weights(ppg_qi_model_weights)
ppg_qi_model._make_predict_function()
pgg_qi_scaler = pickle.load(open(ppg_qi_scaler_file, "rb"))

qi_threshold = 0.5
overwrite = False

train_save_dir = os.path.join(project_dir, "train_windows_ppg_qi")
val_save_dir = os.path.join(project_dir, "val_windows_ppg_qi")
test_save_dir = os.path.join(project_dir, "test_windows_ppg_qi")


with open("filter.log", "w") as log_f:
    for input_dir, save_dir in [(train_dir, train_save_dir), (val_dir, val_save_dir), (test_dir, test_save_dir)]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        total_window_count = 0
        valid_window_count = 0

        data_files = glob.glob(os.path.join(input_dir, "*.npy"))
        print("Found {} files".format(len(data_files)))

        for f in tqdm(data_files):
            base_filename = os.path.splitext(os.path.basename(f))[0]
            # skip this file if it exists and we don't want to overwrite
            if overwrite is False and os.path.exists(os.path.join(save_dir, base_filename + "_v2.npy")):
                continue

            X = np.load(f, allow_pickle=True)
            num_windows = int(X.shape[0] / window_size)

            valid_windows = []
            for i in range(num_windows):
                total_window_count += 1
                idx = int(i * window_size)
                # we use a sliding window to check if we have a valid batch of data
                # (i.e. every window in in sliding window needs to be valid; this possibly
                # can be relaxed using some threshold)
                yy = X[idx:idx + window_size, ppg_col]

                # scale waveform
                yy = pgg_qi_scaler.transform(yy.reshape(-1, 1))

                ppg_qi_pred = ppg_qi_model.predict(np.array([yy, ]))[0][0]
                # if quality over threshold, take the window
                if ppg_qi_pred > qi_threshold:
                    valid_window_count += 1
                    if len(valid_windows) == 0:
                        valid_windows = X[idx:idx + window_size, :]
                    else:
                        valid_windows = np.append(valid_windows, X[idx:idx + window_size, :], axis=0)
            if type(valid_windows) != list:
                np.save(os.path.join(save_dir, base_filename + "_v2.npy"), valid_windows)
        print("{}/{} ({:.2f}%) of the windows in {} were kept".format(valid_window_count, total_window_count,
                                                                      (valid_window_count/total_window_count)*100.,
                                                                      input_dir))
        log_f.write("{}/{} ({:.2f}%) of the windows in {} were kept\n".format(valid_window_count, total_window_count,
                                                                              (valid_window_count/total_window_count)*100.,
                                                                              input_dir))
