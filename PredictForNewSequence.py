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


def main():
    model_name = sys.argv[1]
    sequence = sys.argv[2]
    cs_pos = predict_per_rec(sequence, model_name)
    print 'Predicted cleavage position is %d with model %s' % (cs_pos, model_name)
    return


if __name__ == "__main__":
    main()
