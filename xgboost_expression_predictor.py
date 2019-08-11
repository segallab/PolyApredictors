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


def model_xgboost(params):
    import xgboost as xgb
    model = xgb.XGBRegressor(**params)
    return model


def prepare_input(df):
    xy = df[['Sequence', 'Expression']].copy()
    xy.dropna(inplace=True)
    input_seqs = xy['Sequence'].values.tolist()
    input_seqs_one_hot = one_hot_encoding_flat(np.array(input_seqs))
    y = xy['Expression'].values
    return input_seqs_one_hot, y


def plot_model_fitting(results, file_name, x_min=0):
    epochs = range(1, len(results['validation_0']['rmse']) + 1)
    sns.set(font_scale=3, palette=my_pal)
    fig = plt.figure(figsize=(10, 10))
    sns.regplot(np.array(epochs)[x_min:], np.array(results['validation_0']['rmse'][x_min:]), fit_reg=False, scatter_kws={'s': 100}, label='Training')
    sns.regplot(np.array(epochs)[x_min:], np.array(results['validation_1']['rmse'][x_min:]), fit_reg=False, scatter_kws={'s': 100}, label='Validation')
    plt.xlim((x_min, len(results['validation_0']['rmse']) + 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    fig.tight_layout()
    fig.savefig('Plots/' + file_name + '.png')
    plt.close(fig)
    return


def train_expression_on_full(params):
    model = model_xgboost(params)
    data = pd.read_csv('Supplemental_Table_9.tab', sep='\t', skiprows=1)
    data = data.applymap(lambda x: pd.to_numeric(x, errors='ignore'))  # Make sure that numeric types are numeric
    df_train = data[data.Fold.isin(range(10))]
    df_test = data[data.Fold.isin(['Test'])]
    x_train, y_train = prepare_input(df_train)
    x_test, y_test = prepare_input(df_test)
    eval_set = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train, eval_metric=["rmse"], eval_set=eval_set)
    results = model.evals_result()
    plot_model_fitting(results, 'xgb_loss_all_data')
    y_pred = model.predict(x_test).squeeze()
    plot_scatter_with_correlation(y_test, y_pred, 'xgb_test_scatter')
    return


def main():
    params = {'objective': 'reg:linear', 'n_estimators': 300}
    train_expression_on_full(params)
    return


if __name__ == "__main__":
    main()
