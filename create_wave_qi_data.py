import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import argparse

# order of signals in .npy files
ecg_col = 0
ppg_col = 1
prox_col = 2
nibp_sys_col = 3
nibp_dias_col = 4
nibp_mean_col = 5
abp_col = -1  # should be 6
sample_freq = 100


def main():
    parser = argparse.ArgumentParser(description="Script for generating feature matrices from raw waveform windows")
    parser.add_argument('--data-dir', help='Directory  with input file(s) containing raw windows',
                        default="/Volumes/External/mimic_v9_4s/train_patients", dest="data_dir")
    parser.add_argument('--save-dir', help='Directory to save result files',
                        default="/Volumes/External/ppg_qi/mimic_v9_4s", dest="save_dir")
    parser.add_argument('--signal-type', help="The signal_type to use (either 'ppg', 'abp', or 'ecg')",
                        required=True, dest="signal_type")
    parser.add_argument('--window-length', help="Length of the waveform windows (default=400)",
                        default=400, dest="window_len", type=int)
    parser.add_argument('--num-files', help="Number of files to sample (default=5000)",
                        default=5000, dest="num_files", type=int)
    parser.add_argument('--fig-height', help="Height of matplotlib.pyplot figure",
                        default=8, dest="fig_height", type=int)
    parser.add_argument('--fig-length', help="Length of matplotlib.pyplot figure",
                        default=12, dest="fig_length", type=int)
    args = parser.parse_args()

    signal_type = args.signal_type
    if signal_type not in ["ppg", "abp", "ecg"]:
        raise ValueError("--signal_type must be either 'ppg', 'abp', or 'ecg'")

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise OSError("Data directory does not exist: {}".format(data_dir))
    print("Getting list of numpy arrays...")
    data_files = glob.glob(os.path.join(data_dir, "*.npy"))
    window_len = args.window_len

    save_dir = args.save_dir
    numpy_save_dir = os.path.join(save_dir, "{}_arrays".format(signal_type))
    image_save_dir = os.path.join(save_dir, "{}_images".format(signal_type))

    if not os.path.exists(numpy_save_dir):
        os.makedirs(numpy_save_dir)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    image_file_paths = []

    for f in tqdm(np.random.choice(data_files, size=args.num_files)):
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
            ecg = X[idx:idx + window_len, ecg_col]
            ppg = X[idx:idx + window_len, ppg_col]
            abp = X[idx:idx + window_len, abp_col]
            prox = X[idx:idx + window_len, prox_col]
            nibp_sys = X[idx:idx + window_len, nibp_sys_col]
            nibp_dias = X[idx:idx + window_len, nibp_dias_col]

            if signal_type == "ecg":
                signal = ecg
            elif signal_type == "ppg":
                signal = ppg
            elif signal_type == "abp":
                signal = abp
            else:
                raise ValueError("--signal_type must be either 'ppg', 'abp', or 'ecg'")

            # save numpy array with signal data to file
            numpy_filename = "{}_{}_{}.npy".format(patient_id, file_id, i)
            np.save(os.path.join(numpy_save_dir, numpy_filename), signal)
            # save image of signals to file
            image_filename = "{}_{}_{}.jpg".format(patient_id, file_id, i)
            fig, ax = plt.subplots(4, 1, figsize=(args.fig_length, args.fig_height))
            ax[0].set_title("Sys: {:.0f} Dias: {:.0f} Time: {:.0f}".format(np.median(nibp_sys),
                                                                           np.median(nibp_dias),
                                                                           np.median(prox) / sample_freq))
            ax[0].plot(abp, c='green')
            ax[0].plot(nibp_sys, c='red', linestyle='--')
            ax[0].plot(nibp_dias, c='red', linestyle='--')
            # ax[0].set_ylim([0, 200])
            ax[1].plot(abp[0:int(window_len / 4)], c='green')
            ax[2].plot(ppg, c='orange')
            ax[3].plot(ecg, c='blue')
            image_file_path = os.path.join(image_save_dir, image_filename)
            plt.savefig(image_file_path)
            plt.close()
            image_file_paths.append(os.path.join("http://localhost:8080/static/{}_images".format(signal_type),
                                                 image_filename))

        pd.DataFrame(image_file_paths, columns=["image"]).to_csv(
            os.path.join(save_dir,
                         "{}_image_file_paths.csv".format(signal_type)),
            index=False, header=True)


if __name__ == "__main__":
    main()
