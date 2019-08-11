#! /usr/wisdom/python/bin/python
#$ -S /usr/wisdom/python/bin/python
#$ -V
#$ -cwd
#$ -l mem_free=10g
#$ -pe threads 16
#$ -q himem7.q

import sys
import os
sys.path.append(os.getcwd())
from Common import *

my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                            (35, 180, 169),
                            (246, 242, 17)])/255).values.tolist())


def load_expression_model(model_name):
    from keras.models import load_model
    import cPickle as pickle
    from joblib import load
    if model_name == 'cnn':
        expression_model_path = 'SavedModels/saved_cnn_expression_model.h5'
        expression_model = load_model(expression_model_path)
    elif model_name == 'xgb':
        expression_model_path = 'SavedModels/saved_xgb_expression_model.pkl'
        with open(expression_model_path) as f:
            expression_model = pickle.load(f)
    elif model_name == 'kmer':
        expression_model_path = 'SavedModels/saved_kmer_expression_model.joblib'
        expression_model = load(expression_model_path)
    return expression_model


def prepare_input(seq_fragments, model_name):
    if model_name == 'cnn':
        x = one_hot_encoding_1D(seq_fragments.values)
    elif model_name == 'xgb':
        x = one_hot_encoding_flat(seq_fragments.values)
    elif model_name == 'kmer':
        kmers = get_kmers()
        x = extract_counts(seq_fragments, kmers)
    return x


def predict_per_rec(rec, model_name):
    expression_model = load_expression_model(model_name)
    seq_fragments = pd.Series([rec[i:i + 250] for i in np.arange(0, len(rec) - 250 + 1, 1)])
    x_test = prepare_input(seq_fragments, model_name)
    seq_fragments_predicted = pd.Series(expression_model.predict(x_test).squeeze(),
                                        index=seq_fragments.index).to_frame('Predicted')
    seq_fragments_predicted = seq_fragments_predicted.squeeze()
    cs_pos = seq_fragments_predicted.idxmax() + 145
    return cs_pos


def plot_absolute_differences(cs_true, cs_pred, file_name):
    cs_pos_comparison = pd.merge(cs_true.to_frame(), cs_pred.to_frame('Predicted_CS_pos'),
                                 left_index=True, right_index=True)
    pos_diffs = np.abs(cs_pos_comparison.CS_pos - cs_pos_comparison.Predicted_CS_pos)
    sns.set(font_scale=2.5, palette=my_pal)
    fig = plt.figure(figsize=(10, 8))
    sns.distplot(pos_diffs, kde=False, bins=np.arange(0, 1001, 10))
    plt.xlim((0, 1000))
    ax = plt.gca()
    # ax.set_yticklabels(['%.2f' % (i / len(pos_diffs)) for i in ax.get_yticks()])
    plt.text(0.75, 0.9, 'N=%d' % len(pos_diffs), transform=ax.transAxes, size=25)
    plt.xlabel('Absolute difference between measured\nand predicted cleavage position')
    plt.ylabel("Number of 3\' UTRs")
    fig.tight_layout()
    fig.savefig('Plots/' + file_name + '.png')
    return


def run_per_model(endogenous, model_name):
    cs_predicted = endogenous.Sequence.apply(lambda x: predict_per_rec(x, model_name))
    cs_predicted.to_pickle(model_name + '_endogenous_predicted_cs.pkl')
    plot_absolute_differences(endogenous.CS_pos, cs_predicted, model_name + '_endogenous_cs_absolute_differences')
    return


def main():
    model_names = ['cnn', 'xgb', 'kmer']
    endogenous = pd.read_csv('Supplemental_Table_8.tab', sep='\t', skiprows=1).set_index('GeneName')
    # for model_name in model_names:
    #     run_per_model(endogenous, model_name)
    model_name = model_names[0]
    print model_name
    run_per_model(endogenous, model_name)
    return


if __name__ == "__main__":
    main()
