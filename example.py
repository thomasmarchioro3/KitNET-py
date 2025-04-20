import time
import logging
import zipfile

# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

# Local modules
from src.kitnet import KitNET


if __name__ == "__main__":

    logger = logging.Logger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


    zip_filename = "dataset.zip"
    data_filename = "mirai3.csv"
    timestamp_filename = "mirai3_ts.csv"
    attack_start_index = 71_000

    # Hyperparameters
    max_autoencoder_size = 10
    feature_map_learning_period = 5_000
    training_period = 50_000

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall()

    X = pd.read_csv(data_filename, header=None).values
    timestamps = pd.read_csv(timestamp_filename,header=None).values
    logger.log(level=logging.INFO, msg="Reading sample dataset...")

    model = KitNET(X.shape[1], max_autoencoder_size, training_period, feature_map_learning_period)

    errors = np.zeros(X.shape[0])

    logger.log(level=logging.INFO, msg="Running KitNET:")

    tic = time.perf_counter()
    for i in range(X.shape[0]):
        errors[i] = model.process(X[i])
        if (i % 1000) == 0:
            logger.log(level=logging.INFO, msg=f"Processed {i:d} samples. Error on current sample: {errors[i]:.4f}.")

    toc = time.perf_counter()
    logger.log(level=logging.INFO, msg=f"Complete. Time elapsed: {toc - tic}")

    benign_examples = np.log(
        errors[feature_map_learning_period+training_period+1:attack_start_index]
    )

    log_probs = norm.logsf(np.log(errors), np.mean(benign_examples), np.std(benign_examples))

    plt.figure(figsize=(10,5))
    fig = plt.scatter(
        timestamps[feature_map_learning_period+training_period+1:],
        errors[feature_map_learning_period+training_period+1:],
        s=0.1,
        c=log_probs[feature_map_learning_period+training_period+1:],
        cmap='RdYlGn'
    )
    plt.yscale("log")
    plt.title("Anomaly Scores from KitNET's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Time elapsed [min]")
    plt.annotate(
        'Mirai C&C channel opened [Telnet]', 
        xy=(timestamps[71662],errors[71662]), 
        xytext=(timestamps[58000],1),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )
    plt.annotate(
        'Mirai Bot Activated\nMirai scans network for vulnerable devices', 
        xy=(timestamps[72662],1), 
        xytext=(timestamps[55000],5),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )
    figbar=plt.colorbar()
    figbar.ax.set_ylabel('Log Probability\n ', rotation=270, labelpad=10)
    plt.show()
