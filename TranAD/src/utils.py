import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd 
import numpy as np

class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_accuracies(accuracy_list, folder, lr):
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    plt.figure(figsize=(8,6))

    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Average Training Loss')
    l1, = ax1.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')
    l2, = ax2.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')

    plt.title(f"{folder} training with starting learning_rate={lr}")

    # Combine legends from both axes
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.savefig(f'plots/{folder}/training-graph_lr_{lr}.pdf')
    plt.close(fig)

def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def getresults2(df, result):
	results2, df1, df2 = {}, df.sum(), df.mean()
	for a in ['FN', 'FP', 'TP', 'TN']:
		results2[a] = df1[a]
	for a in ['precision', 'recall']:
		results2[a] = df2[a]
	results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
	return results2