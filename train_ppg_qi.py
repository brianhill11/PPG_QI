import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import json
import shutil
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten
from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib as mpl
import platform

if platform.system() == 'Darwin':
    mpl.use('MacOSX')

project_dir = "/Volumes/External/ppg_qi/mimic_v9_4s"
data_dir = os.path.join(project_dir, "ppg_arrays")
image_dir = os.path.join(project_dir, "ppg_images")
train_split = 0.7
window_size = 400
batch_size = 32

load_scaler_pickle = False
load_weights = False

if not os.path.exists("weights"):
    os.makedirs("weights")
model_file = os.path.join("weights", "weights.11.hdf5")


def get_file_from_url(url):
    """
    Should return p058128-2176-08-29-18-04_135_4.jpg
     from http://localhost:8080/static/ppg_images/p058128-2176-08-29-18-04_135_4.jpg
    """
    return url.split("/")[-1]


def sort_image_files_by_label(label_df, image_dir, save_dir):
    """
    Sort the waveform images files into separate directories based on classification label
    label_df: dataframe containing image name and classification label
    image_dir: directory containing the waveform image files
    save_dir: directory to save the sorted images
    """
    for c in label_df["ppg"].unique():
        save_dir_class = os.path.join(save_dir, c)
        # create new directory to store images of class c
        os.makedirs(save_dir_class, exist_ok=True)
        # get all of the images belonging to class c
        images_to_copy = label_df[label_df["ppg"] == c]["image_file"]
        print("Copying {} {} images...".format(images_to_copy.shape[0], c))
        # copy these images to their class directory
        for image in images_to_copy:
            shutil.copy2(os.path.join(image_dir, image), os.path.join(save_dir_class, image))


def ci(values, alpha=0.95):
    # if we get lower value, flip
    if alpha < 0.5:
        alpha = 1. - alpha
    interp = "lower"
    return np.quantile(values, 1.-alpha, interpolation=interp), np.quantile(values, alpha, interpolation=interp)


class DataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, data_dir, patients, label_file,
                 window_len=400, batch_size=32, step_size=1, shuffle=True, X_scaler=None):
        """Initialization"""
        self.data_dir = data_dir
        self.window_len = window_len  # number of samples to use in window
        self.batch_size = batch_size  # number of windows to use in sliding window
        self.step_size = step_size  # number of samples between windows in sliding window
        self.shuffle = shuffle  # if shuffle, don't use sliding window for batches
        self.data_files = os.listdir(self.data_dir)
        self.num_windows = 0
        self.file_count = 0
        self.patients = patients
        self.label_file = label_file

        # for each file, we need the training label
        def read_label_json(jsn):
            try:
                return int(json.loads(jsn)["quality"])
            except ValueError:
                if json.loads(jsn)["quality"] == "nan":
                    return np.nan
                else:
                    print("ERROR: found expected label: {}".format(json.loads(jsn)["quality"]))
                    exit()

        def create_label(value):
            if value in ["valid"]:
                return 1
            elif value in ["invalid", "neutral"]:
                return 0
            else:
                raise ValueError("Label value must either be 'valid', 'neutral', or 'invalid'")

        self.label_df = pd.read_csv(self.label_file, sep=",", header=0)
        # get labels for each file
        self.label_df["label"] = self.label_df["ppg"].apply(create_label)
        self.label_df = self.label_df[~self.label_df["label"].isna()]
        # get the file prefix
        self.label_df["file_prefix"] = self.label_df["image"].apply(
            lambda fn: os.path.splitext(get_file_from_url(fn))[0])
        # use the file prefix to get the corresponding numpy array file
        self.label_df["array_file"] = self.label_df["file_prefix"] + ".npy"
        # extract the patient ID
        self.label_df["patient_ids"] = self.label_df["image"].apply(lambda fn: get_file_from_url(fn).split("-")[0])
        # only use rows where the data is in our valid patient list
        self.label_df = self.label_df[self.label_df["patient_ids"].isin(self.patients)]
        print("Found {} valid files".format(self.label_df.shape[0]))
        print("Label 1 frequency: {:.2f}%".format((self.label_df["label"] > 0).mean() * 100.))
        print(self.label_df["label"].value_counts())

        # # get list of files to use for this generator object
        self.patient_files = [os.path.join(data_dir, x) for x in self.label_df["array_file"].values]
        # for p in self.patients:
        #     self.patient_files = self.patient_files + glob.glob(os.path.join(self.data_dir, p + "*.npy"))
        # print("Found {} files for generator".format(len(self.patient_files)))

        # if this is for training data, we need to fit scalers
        if X_scaler is None:
            print("initializing new StandardScaler objects")
            self.X_scaler = StandardScaler()
        # otherwise, for test we initialize to training scalers
        else:
            print("using supplied StandardScaler objects")
            self.X_scaler = X_scaler
            return
        for f in tqdm(self.patient_files):
            self.file_count += 1
            if self.file_count % 5000 == 0:
                pickle.dump(self.X_scaler, open("weights/ppg_qi_scaler.pkl", 'wb'))
            #         for f in self.data_files:
            x = np.load(os.path.join(f), allow_pickle=True)
            #             print(X.shape)
            num_windows = int(x.shape[0] / self.window_len)
            self.num_windows += num_windows
            if X_scaler is None:
                # change here for ECG + SpO2
                self.X_scaler.partial_fit(x.reshape(-1, 1))

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return max(int(self.label_df.shape[0] / self.batch_size), 1)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []

        if self.shuffle:
            np.random.shuffle(self.patient_files)

        i = index * self.batch_size
        # files to be read in this batch
        batch_files = self.patient_files[i:i + batch_size]
        # for each file, read numpy array, scale, and get corresponding label
        for f in batch_files:
            x = np.load(f)
            x = self.X_scaler.transform(x.reshape(-1, 1))
            batch_x.append(x)

            # get corresponding label
            base_filename = os.path.splitext(os.path.basename(f))[0]
            y = self.label_df[self.label_df["file_prefix"] == base_filename]["label"].values[0]
            y = 1 if y == 1 else 0
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y)
        # return np.array(batch_x), np.array(to_categorical(batch_y, num_classes=2))


def create_model():
    num_filters = 64
    trainable = True
    optimizer = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=0.5)

    inputs = Input(batch_shape=(batch_size, window_size, 1))

    for i in range(3):
        if i == 0:
            output = Conv1D(num_filters, kernel_size=15, activation=None,
                            input_shape=(batch_size, window_size, 1),
                            trainable=trainable)(inputs)
        else:
            output = Conv1D(num_filters, kernel_size=9, activation=None,
                            input_shape=(batch_size, window_size, 1),
                            trainable=trainable)(output)
        output = BatchNormalization()(output)
        output = ReLU()(output)
        output = MaxPool1D()(output)

    output = Flatten()(output)
    # output = BatchNormalization()(output)
    # output = ReLU()(output)
    output = Dense(1, activation="sigmoid", trainable=True)(output)

    model = Model(inputs=inputs, outputs=[output])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc',
                           tf.keras.metrics.AUC(curve='ROC', name="roc"),
                           tf.keras.metrics.AUC(curve='PR', name="pr")])
    model.summary()

    os.makedirs("figures", exist_ok=True)
    plot_model(model, to_file=os.path.join("figures", "model.png"), show_shapes=True)
    return model


if __name__ == '__main__':
    label_file = os.path.join(project_dir, "result.csv")
    label_df = pd.read_csv(label_file, sep=",", header=0)

    # create separate directories to exclusively store images from each class
    label_df["image_file"] = label_df["image"].apply(lambda fn: get_file_from_url(fn))
    sort_image_files_by_label(label_df, image_dir, os.path.join(project_dir, "images_by_class"))

    label_df["patient_ids"] = label_df["image"].apply(lambda fn: get_file_from_url(fn).split("-")[0])
    valid_patients = label_df["patient_ids"].unique()

    print("Label distribution:")
    print(label_df["ppg"].value_counts(normalize=True))
    print("Window per patient distribution:")
    print(label_df.groupby("patient_ids").apply(lambda x: x.shape[0]).describe(include="all"))

    patient_list = label_df["patient_ids"].unique()
    print("Found {} valid patients with labeled data".format(len(patient_list)))
    train_patients, test_patients = train_test_split(patient_list, train_size=train_split)
    print("Training on {} patients, testing on {} patients".format(len(train_patients), len(test_patients)))

    # create DataGenerator objects
    # optionally load existing scaler objects
    if not load_scaler_pickle:
        train_gen = DataGenerator(data_dir=data_dir,
                                  patients=train_patients,
                                  label_file=label_file,
                                  window_len=window_size,
                                  batch_size=batch_size)
        pickle.dump(train_gen.X_scaler, open("weights/ppg_qi_scaler.pkl", "wb"))
    else:
        X_scaler = pickle.load(open("weights/ppg_qi_scaler.pkl", "rb"))
        train_gen = DataGenerator(data_dir=data_dir,
                                  patients=train_patients,
                                  label_file=label_file,
                                  window_len=window_size,
                                  batch_size=batch_size,
                                  X_scaler=X_scaler)

    # use mean/stdev from training data to scale testing data
    val_gen = DataGenerator(data_dir=data_dir,
                            patients=test_patients,
                            label_file=label_file,
                            window_len=window_size,
                            batch_size=batch_size,
                            X_scaler=train_gen.X_scaler)

    print("train generator has {} batches".format(train_gen.__len__()))
    print(train_gen.__getitem__(0)[0].shape)

    checkpoint = ModelCheckpoint(os.path.join("weights", "weights.{epoch:02d}.hdf5"),
                                 monitor='val_roc',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='max',
                                 save_freq='epoch')

    early_stop = EarlyStopping(monitor='val_roc', patience=5, verbose=1, mode='max')

    # train the model
    if load_weights:
        print("Loading model...")
        model = tf.keras.models.load_model(model_file)
        print("Model loaded.")
    else:
        model = create_model()

    history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        # validation_steps=50,
                        # steps_per_epoch=1000,
                        epochs=80,
                        verbose=1,
                        callbacks=[checkpoint, early_stop],
                        initial_epoch=0,
                        use_multiprocessing=False,
                        max_queue_size=500,
                        workers=1)

    X, y = val_gen.__getitem__(0)
    y_pred = model.predict_on_batch(X)
    # for i in range(X.shape[0]):
    #     print("i: {} y_true: {} y_pred: {}".format(str(i), y[i], y_pred[i]))
    # for i in range(X.shape[0]):
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #     ax.plot(X[i])
    #     ax.set_title("i: {} y_true: {} y_pred: {}".format(str(i), y[i], y_pred[i]))
    #     plt.show()

    val_gen.shuffle = False
    val_true = []
    val_preds = []
    for i in range(len(val_gen)):
        X, y = val_gen.__getitem__(i)
        for l in y:
            val_true.append(l)
        val_pred = model.predict_on_batch(X)
        for p in val_pred:
            val_preds.append(p)

    # make ROC plot
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fpr, tpr, thresholds = roc_curve(val_true, val_preds)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    lw = 2
    fontsize = 14
    ax[0].plot(fpr, tpr, color='darkorange',
               lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(val_true, val_preds))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].set_xlabel('False Positive Rate', fontsize=fontsize)
    ax[0].set_ylabel('True Positive Rate', fontsize=fontsize)
    ax[0].set_title('ROC Curve', fontsize=fontsize)
    ax[0].legend(loc="lower right", fontsize=fontsize)

    # plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(val_true, val_preds)
    ax[1].plot(recall, precision, marker='.', lw=lw,
               label='P-R curve (area = %0.2f)' % average_precision_score(val_true, val_preds))
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    # axis labels
    ax[1].set_xlabel('Recall', fontsize=fontsize)
    ax[1].set_ylabel('Precision', fontsize=fontsize)
    ax[1].set_title('Precision-Recall Curve', fontsize=fontsize)
    # show the legend
    ax[1].legend(loc="lower right", fontsize=fontsize)

    plt.savefig(os.path.join("figures", "roc_pr_curves.png"))
    plt.show()

    print("=" * 40)
    precision_threshold = 0.95
    # get minimum threshold where precision is >= precision_threshold
    threshold = np.min(thresholds[np.where(precision[:-1] >= precision_threshold)])
    print("threshold for precision of {}: {}".format(precision_threshold, threshold))

    # calculate bootstrapped metrics
    roc_auc_bs = []
    pr_auc_bs = []
    precision_bc = []
    recall_bc = []
    num_bootstrap = 1000
    for _ in range(num_bootstrap):
        bootstrap_idx = np.random.choice(np.arange(len(val_true)), size=len(val_true), replace=True)
        val_true_bs = np.array(val_true)[bootstrap_idx]
        val_prob_bs = np.array(val_preds)[bootstrap_idx]
        roc_auc_bs.append(roc_auc_score(val_true_bs, val_prob_bs))
        pr_auc_bs.append(average_precision_score(val_true_bs, val_prob_bs))
        val_pred_bs = [1 if x > threshold else 0 for x in val_prob_bs]
        precision_bc.append(precision_score(val_true_bs, val_pred_bs))
        recall_bc.append(recall_score(val_true_bs, val_pred_bs))

    print("ROC AUC: {:.3f} ({:.3f}-{:.3f})".format(np.mean(roc_auc_bs), *ci(roc_auc_bs)))
    print("PR AUC: {:.3f} ({:.3f}-{:.3f})".format(np.mean(pr_auc_bs), *ci(pr_auc_bs)))
    print("Precision: {:.3f} ({:.3f}-{:.3f})".format(np.mean(precision_bc), *ci(precision_bc)))
    print("Recall: {:.3f} ({:.3f}-{:.3f})".format(np.mean(recall_bc), *ci(recall_bc)))
    print("=" * 40)
    print("Precision Recall F-score Support:",
          precision_recall_fscore_support(val_true,
                                          [1 if x > threshold else 0 for x in val_preds],
                                          pos_label=1,
                                          average='binary'))
