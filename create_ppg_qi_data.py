import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import glob

# order of signals in .npy files
ecg_col = 0
ppg_col = 1
prox_col = 2
nibp_sys_col = 3
nibp_dias_col = 4
nibp_mean_col = 5
abp_col = -1  # should be 6

data_dir = "/Volumes/External/mimic_v9_4s/train_patients"
data_files = glob.glob(os.path.join(data_dir, "*.npy"))
window_len = 400

save_dir = "/Volumes/External/ppg_qi/mimic_v9_4s"
numpy_save_dir = os.path.join(save_dir, "ppg_arrays")
image_save_dir = os.path.join(save_dir, "ppg_images")

if not os.path.exists(numpy_save_dir):
    os.makedirs(numpy_save_dir)
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

image_file_paths = []

for f in tqdm(np.random.choice(data_files, size=5000)):
    patient_id = os.path.basename(f).split("_")[0]
    file_id = os.path.basename(f).split("_")[1]
    X = np.load(os.path.join(data_dir, f), allow_pickle=True)
    num_windows = int(X.shape[0] / window_len)

    # choose a random spot in the waveform
    # for i in range(num_windows):
    for i in range(5):
        # choose a random spot in the waveform
        val = np.random.randint(low=0, high=num_windows)
        idx = int(val * window_len)
        # for each window in file
        # idx = int(i * window_len)

        # we use a sliding window to check if we have a valid batch of data
        # (i.e. every window in in sliding window needs to be valid; this possibly
        # can be relaxed using some threshold)
        ekg = X[idx:idx + window_len, ecg_col]
        ppg = X[idx:idx + window_len, ppg_col]
        abp = X[idx:idx + window_len, abp_col]

        # save numpy array with ppg data to file
        numpy_filename = "{}_{}_{}.npy".format(patient_id, file_id, i)
        np.save(os.path.join(numpy_save_dir, numpy_filename), ppg)
        # save image of ppg signal to file
        image_filename = "{}_{}_{}.jpg".format(patient_id, file_id, i)
        fig, ax = plt.subplots(3, 1, figsize=(12, 8))
        ax[0].plot(abp, c='green')
        # ax2 = ax.twinx()
        ax[1].plot(ppg, c='orange')
        image_file_path = os.path.join(image_save_dir, image_filename)
        plt.savefig(image_file_path)
        plt.close()
        image_file_paths.append(os.path.join("http://localhost:8080/static/ppg_images", image_filename))

pd.DataFrame(image_file_paths, columns=["image"]).to_csv(os.path.join(save_dir, "image_file_paths.csv"),
                                                         index=False, header=True)
