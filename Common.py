from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_scatter_with_correlation(y_true, y_pred, file_name):
    from scipy import stats
    my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                                              (35, 180, 169),
                                              (246, 242, 17)]) / 255).values.tolist())
    sns.set(font_scale=3, palette=my_pal)
    fig = plt.figure(figsize=(10, 10))
    ax = sns.regplot(y_true, y_pred, fit_reg=False)
    plt.xlabel('Measured expression [log2]')
    plt.ylabel('Predicted expression [log2]')
    plt.xlim((int(y_true.min()) - 1, int(y_true.max() + 1)))
    plt.ylim((int(y_pred.min() - 1), int(y_pred.max() + 1)))
    val = stats.pearsonr(y_true, y_pred)
    if val[1] < 10 ** -10:
        plt.text(0.1, 0.8, 'R=%.2f\n$p=<10^{-10}$\nN=%d' % (val[0], len(y_true)),
                 transform=ax.transAxes, size=30)
    else:
        plt.text(0.1, 0.8, 'R=%.2f\np=%.2E\nN=%d' % (val[0], val[1], len(y_true)),
                 transform=ax.transAxes, size=30)
    fig.tight_layout()
    fig.savefig('Plots/' + file_name+'.png')
    plt.close(fig)
    return


def plot_model_fitting(history, file_name, x_min=0):
    my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                                              (35, 180, 169),
                                              (246, 242, 17)]) / 255).values.tolist())
    epochs = range(1, len(history['loss'])+1)
    sns.set(font_scale=3, palette=my_pal)
    fig = plt.figure(figsize=(10, 10))
    sns.regplot(np.array(epochs)[x_min:], np.array(history['loss'][x_min:]), fit_reg=False, scatter_kws={'s': 100}, label='Training')
    sns.regplot(np.array(epochs)[x_min:], np.array(history['val_loss'])[x_min:], fit_reg=False, scatter_kws={'s': 100}, label='Validation')
    plt.xlim((x_min, len(history['loss'])+1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    fig.tight_layout()
    fig.savefig('Plots/' + file_name + '.png')
    plt.close(fig)
    return


def one_hot_encoding_1D(sequences):
    """Perform one hot encoding on DNA sequences.
    sequences is a list of DNA sequences.
    Returns a numpy array of the shape (number_of_sequences, max_len, 4).
    This is compatible as input for 1D CNN."""
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(['ACGT'])
    sequence_of_int = tokenizer.texts_to_sequences(sequences)
    one_hot_encoded = to_categorical(sequence_of_int)
    one_hot_encoded = one_hot_encoded[:, 1:]
    one_hot_reshaped = one_hot_encoded.reshape((sequences.shape[0], 250, 4))
    return one_hot_reshaped


def one_hot_encoding_flat(sequences):
    """Perform one hot encoding on DNA sequences.
    sequences is a list of DNA sequences.
    Returns a numpy array of the shape (number_of_sequences, max_len, 4).
    This is compatible as input for 1D CNN."""
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(['ACGT'])
    sequence_of_int = tokenizer.texts_to_sequences(sequences)
    one_hot_encoded = to_categorical(sequence_of_int)
    one_hot_encoded = one_hot_encoded[:, 1:]
    one_hot_reshaped = one_hot_encoded.reshape((sequences.shape[0], 250*4))
    return one_hot_reshaped


def get_kmer_list(k):
    import itertools
    kmers = []
    for i in itertools.product(['A', 'C', 'G', 'T'], repeat=k):
        kmers.append(''.join(list(i)))
    return kmers


def get_kmers():
    kmers_all = [get_kmer_list(k) for k in np.arange(1, 7, 1)]
    kmers_all = [item for sublist in kmers_all for item in sublist]
    return kmers_all


def occurrences(string, sub):
    count = 0
    start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def extract_counts(seqs, kmers):
    counts = seqs.apply(lambda x: pd.Series(dict([(kmer, occurrences(x, kmer)) for kmer in kmers])))
    return counts


def extract_kmer_features(sequences):
    kmers = get_kmers()
    kmer_featured = extract_counts(sequences, kmers)
    return kmer_featured
