#! /usr/wisdom/python/bin/python
#$ -S /usr/wisdom/python/bin/python
#$ -V
#$ -cwd
#$ -l mem_free=10g
#$ -pe threads 2
#$ -q himem7.q

import sys
import os
sys.path.append(os.getcwd())
from Common import *

n_threads = 2  # Number of threads for keras to use
my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                            (35, 180, 169),
                            (246, 242, 17)])/255).values.tolist())


def expression_model(params):
    from keras import layers
    from keras import models
    from keras.optimizers import Adam
    from keras.regularizers import l1_l2
    num_filters = params['num_filters']
    num_filters2 = params['num_filters2']
    kernel_size = params['kernel_size']
    kernel_size2 = params['kernel_size2']
    l1_lambda = params['l1_lambda']
    l2_lambda = params['l2_lambda']
    lr = params['lr']
    model = models.Sequential()
    model.add(layers.Conv1D(num_filters, kernel_size, input_shape=(250, 4), activation='relu',
                            kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), name='1st_Conv1D'))
    model.add(layers.MaxPooling1D(2, strides=1, name='1st_MaxPooling1D'))
    model.add(layers.Dropout(0.5, name='1st_Dropout'))
    model.add(layers.Conv1D(num_filters2, kernel_size2, activation='relu',
                            kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), name='2nd_Conv1D'))
    model.add(layers.MaxPooling1D(2, strides=1, name='2nd_MaxPooling1D'))
    model.add(layers.Dropout(0.5, name='2nd_Dropout'))
    model.add(layers.Flatten(name='Flatten'))
    model.add(layers.Dense(1, kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), name='Output'))
    model.compile(Adam(lr=lr), 'mse')
    return model


def prepare_input(df):
    xy = df[['Sequence', 'Expression']].copy()
    xy.dropna(inplace=True)
    input_seqs = xy['Sequence'].values.tolist()
    input_seqs_one_hot = one_hot_encoding_1D(np.array(input_seqs))
    y = xy['Expression'].values
    return input_seqs_one_hot, y


def plot_model_fitting(history, file_name, x_min=0):
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


def train_expression_on_full(params):
    from keras import backend as K
    K.set_session(
        K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=n_threads,
                                             inter_op_parallelism_threads=n_threads)))
    model = expression_model(params)
    data = pd.read_csv('Supplementary_Table_9.tab', sep='\t')
    data = data.applymap(lambda x: pd.to_numeric(x, errors='ignore'))  # Make sure that numeric types are numeric
    df_train = data[data.Fold.isin(range(10))]
    df_test = data[data.Fold.isin(['Test'])]
    x_train, y_train = prepare_input(df_train)
    x_test, y_test = prepare_input(df_test)
    epochs = params['epochs']
    batch_size = params['batch_size']
    fit_params = {'x': x_train, 'y': y_train, 'epochs': epochs, 'batch_size': batch_size,
                  'validation_data': (x_test, y_test)}  # Validation data is only used to look at loss curves'
    history = model.fit(**fit_params)
    plot_model_fitting(history.history, 'cnn_loss_all_data')
    y_pred = model.predict(x_test).squeeze()
    plot_scatter_with_correlation(y_test, y_pred, 'cnn_test_scatter')
    return


def main():
    params = {'batch_size': 256,
              'epochs': 75,
              'kernel_size': 8,
              'kernel_size2': 6,
              'l1_lambda': 0.0001,
              'l2_lambda': 0.0001,
              'lr': 0.0015,
              'num_filters': 64,
              'num_filters2': 32}
    train_expression_on_full(params)
    return


if __name__ == "__main__":
    main()
