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

my_pal = sns.color_palette((pd.DataFrame([(54, 51, 154),
                            (35, 180, 169),
                            (246, 242, 17)])/255).values.tolist())


def model_kmer_elastic_net(params):
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(**params)
    return model


def prepare_input(df):
    xy = df[['Sequence', 'Expression']].copy()
    xy.dropna(inplace=True)
    input_seqs = xy['Sequence']
    kmer_featured = extract_kmer_features(input_seqs)
    y = xy['Expression'].values
    return kmer_featured, y


def train_expression_on_full(params):
    data = pd.read_csv('Supplemental_Table_9.tab', sep='\t', skiprows=1)
    data = data.applymap(lambda x: pd.to_numeric(x, errors='ignore'))  # Make sure that numeric types are numeric
    df_train = data[data.Fold.isin(range(10))]
    df_test = data[data.Fold.isin(['Test'])]
    x_train, y_train = prepare_input(df_train)
    x_test, y_test = prepare_input(df_test)
    model = model_kmer_elastic_net(params)
    model.fit(x_train, y_train)
    y_pred = pd.Series(model.predict(x_test).squeeze())
    plot_scatter_with_correlation(y_test, y_pred, 'kmer_test_scatter')
    return


def main():
    params = {'alpha': 0.021544346900318822, 'l1_ratio': 0.4}
    train_expression_on_full(params)
    return


if __name__ == "__main__":
    main()
