import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def create_mnist_dataset():
    return mnist.load_data()

def create_toy_regression_dataset(xmin=-10., xmax=10, noise_std=.2):
    x_trn = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)
    y_trn = np.sin(x_trn) + np.random.normal(0, noise_std, size=1000).astype(np.float32)

    x_gt = np.linspace(xmin, xmax, 1000, dtype=np.float32)
    y_gt = np.sin(x_gt)

    x_tst = np.linspace(xmin, xmax, 1000, dtype=np.float32)
    y_tst = np.sin(x_tst) + np.random.normal(0, noise_std, size=1000).astype(np.float32)
    return x_gt, y_gt, x_trn, y_trn, x_tst, y_tst

def plot_regression_model_analysis(gt=None,  trn=None, tst=None,
                                   pred=None, xlim=None, ylim=None,
                                   title=None):
    if gt:
        x_gt, y_gt = gt
        plt.plot(x_gt, y_gt, c='#F0AA00', 
                 alpha=.8, lw=2, label="ground truth")
    if trn:
        x_trn, y_trn = trn
        plt.scatter(x_trn,  y_trn,
                    s=8, ec='black', lw=1, fc=None, alpha=1,
                    label='train samples')
    
    if tst:
        x_tst, y_tst = tst
        plt.scatter(x_tst, y_tst, s=5, c='blue', alpha=.1, label='test samples')

    if pred:
        x_tst, yhat_mean, yhat_std = pred
        plt.scatter(x_tst, yhat_mean, s=5, c='magenta', alpha=1, label='preds')
        if yhat_std is not None:
            plt.fill_between(x_tst, (yhat_mean - 1.*yhat_std), (yhat_mean + 1.*yhat_std), lw=1,
                             ec='blue', fc='blue', alpha=.3, label='preds 1*std')
            plt.fill_between(x_tst, (yhat_mean - 2.*yhat_std), (yhat_mean + 2.*yhat_std), lw=1,
                             ec='blue', fc='blue', alpha=.2, label='preds 2*std')

    if xlim:
        plt.xlim(*xlim)
        
    if ylim:
        plt.ylim(*ylim)
    
    if title:
        plt.title(title)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.35, 1.03), loc='upper right', fancybox=False, framealpha=1.0)

def plot_probabilities(probs):
    plt.bar(np.arange(probs.shape[0]), probs)
    plt.xticks(np.arange(probs.shape[0]))
    plt.xlabel('category')
    plt.ylabel('probability')
    

def plot_calibration_plot(probs, labels, bins=10):
    predicted_label = np.argmax(probs, axis=-1)
    predicted_score = np.max(probs, axis=-1)
    correct_prediction = predicted_label == label
        
    step_size = 1.0 / bins
    
    mean_probabilities = []
    fraction_correct = []
    
    for i in range(bins):
        beg = i * step_size
        end = start + step_size
        mask = (predicted_scores > beg) & (predicted_scores < end) 
        cnt = mask.astype(np.float32).sum()
        correct = (label[mask] == predicted_label[mask]).astype(np.float32).sum()
        
        mean_probabilities.append((beg+end)/2.)
        fraction_crrect.append((correct + 1e-10)/(cnt + 1e-10))
        
    return mean_probabilities, fraction_correct
        
    
    

def multiclass_calibration_curve(probs, labels, bins=10):
    step_size = 1.0 / bins
    n_classes = probs.shape[1]
    labels_ohe = np.eye(n_classes)[labels.astype(np.int64)]

    midpoints = []
    mean_confidences = []
    accuracies = []
    
    for i in range(bins):
        beg = i * step_size
        end = (i + 1) * step_size
        
        bin_mask = (probs >= beg) & (probs < end)
        bin_cnt = bin_mask.astype(np.float32).sum()
        bin_confs = probs[bin_mask]
        bin_acc = labels_ohe[bin_mask].sum() / bin_cnt

        midpoints.append((beg+end)/2.)
        mean_confidences.append(np.mean(bin_confs))
        accuracies.append(bin_acc)
        
    return midpoints, accuracies, mean_confidences

def plot_multiclass_calibration_curve(probs, labels, bins=10, title=None):
    title = 'Reliability Diagram' if title is None else title
    midpoints, accuracies, mean_confidences = multiclass_calibration_curve(probs, labels, bins=bins)
    plt.bar(midpoints, accuracies, width=1.0/float(bins), align='center', lw=1, ec='#000000', fc='#2233aa', alpha=1, label='Model', zorder=0)
    plt.scatter(midpoints, accuracies, lw=2, ec='black', fc="#ffffff", zorder=2)
    plt.plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2, alpha=.7, color='gray', label='Perfectly calibrated', zorder=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('\nconfidence')
    plt.ylabel('accuracy\n')
    plt.title(title+'\n')
    plt.xticks(midpoints, rotation=-45)
    plt.legend(loc='upper left')
    return midpoints, accuracies, mean_confidences