#! /usr/wisdom/python/bin/python
#$ -S /usr/wisdom/python/bin/python
#$ -V
#$ -cwd
#$ -l mem_free=10g
#$ -pe threads 32
#$ -q himem7.q

import sys
import os
sys.path.append(os.getcwd())
from Common import *


n_threads = 32  # Number of threads for keras to use
my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                            (35, 180, 169),
                            (246, 242, 17)])/255).values.tolist())


def cleavage_model(params):
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
    act_l1_lambda = params['act_l1_lambda']
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
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda), name='Dense'))
    model.add(layers.Dense(189, activation='relu', kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda),
                           activity_regularizer=l1_l2(l1=act_l1_lambda, l2=0), name='Output'))
    model.compile(Adam(lr=lr), 'poisson')
    return model


def prepare_input(df):
    output_label = np.arange(41, 230).astype(str).tolist()
    df.dropna(inplace=True)  # drop variants with no expression
    xy = df[['Sequence'] + output_label].copy()
    y = xy[output_label].astype(np.float64)
    input_seqs = xy['Sequence'].values.tolist()
    input_seqs_one_hot = one_hot_encoding_1D(np.array(input_seqs))
    return input_seqs_one_hot, y.values


def plot_cleavage_distirbution_comparison(true_normed, pred_normed, file_name):
    from matplotlib.ticker import MaxNLocator
    sns.set(font_scale=3, palette=my_pal)
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 1, 1)
    plt.bar(np.arange(41, 230, 1), true_normed.mean(), linewidth=0, width=1)
    plt.xticks([41, 75, 110, 145, 180, 215], [])
    plt.xlim(41, 230)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    plt.title('Measured')
    fig.add_subplot(2, 1, 2)
    plt.bar(np.arange(41, 230, 1), pred_normed.mean(), linewidth=0, width=1)
    plt.xticks([41, 75, 110, 145, 180, 215])
    plt.xlim(41, 230)
    plt.xlabel('Position')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    plt.title('Predicted')
    fig.text(0, 0.5, 'Mean normalized\ncleavage efficiency', ha='center', va='center', rotation='vertical')
    fig.tight_layout()
    fig.savefig('Plots/' + file_name + '.png', bbox_inches='tight')
    return


def plot_absolute_differences(true_normed, pred_normed, file_name):
    pos_true = true_normed.idxmax(1)
    pos_pred = pred_normed.idxmax(1)
    combined_pos_preds = pd.concat([pos_true, pos_pred], axis=1)
    combined_pos_preds.columns = ['Measured', 'Predicted']
    filtered_combined_pos_preds = combined_pos_preds[(combined_pos_preds[['Measured', 'Predicted']] > 0).all(1)]  # use only with sequences that had measured and predicted CS
    pos_diffs = filtered_combined_pos_preds.Measured - filtered_combined_pos_preds.Predicted
    sns.set(font_scale=2.5, palette=my_pal)
    fig = plt.figure(figsize=(10, 8))
    sns.distplot(np.abs(pos_diffs), kde=False, bins=np.arange(0, 41, 3))
    ax = plt.gca()
    plt.text(0.75, 0.9, 'N=%d' % len(pos_diffs), transform=ax.transAxes, size=25)
    plt.xlabel('Absolute difference between measured\nand predicted cleavage position')
    plt.ylabel('Number of constructs')
    fig.tight_layout()
    fig.savefig('Plots/' + file_name + '.png', bbox_inches='tight')
    return


def train_cleavage_on_full(params):
    from keras import backend as K
    K.set_session(
        K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=n_threads,
                                             inter_op_parallelism_threads=n_threads)))
    model = cleavage_model(params)
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
    plot_model_fitting(history.history, 'Cleavage_loss_all_data')
    y_pred = model.predict(x_test).squeeze()
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    # y_test.to_pickle('cleavage_test_measured.pkl')
    # y_pred.to_pickle('cleavage_test_predicted.pkl')
    true_normed = pd.DataFrame(y_test).apply(lambda x: x / x.sum(), axis=1).fillna(0)
    pred_normed = pd.DataFrame(y_pred).apply(lambda x: x / x.sum(), axis=1).fillna(0)
    plot_cleavage_distirbution_comparison(true_normed, pred_normed, 'Cleavage_distribution_comparison_test')
    plot_absolute_differences(true_normed, pred_normed, 'Cleavage_absolute_differences_test')
    return


def main():
    params = {'act_l1_lambda': 1e-06,
              'batch_size': 512,
              'epochs': 100,
              'kernel_size': 12,
              'kernel_size2': 8,
              'l1_lambda': 0.0001,
              'l2_lambda': 0.0001,
              'lr': 0.0015,
              'num_filters': 128,
              'num_filters2': 64}
    train_cleavage_on_full(params)
    return


if __name__ == "__main__":
    main()
