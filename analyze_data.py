import ast
import os

import pandas as pd


def main():
    data_dir = './ml_results'
    data_file = 'spec_feature_search.csv'
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path, index_col=0)
    average_df = get_average_df(df)
    top_features_list = get_top_mape_features(average_df, 5)
    target_df = df[df['Input'].isin(top_features_list)]
    get_top_mape_data(target_df)
    return


def get_top_mape_features(df, n):
    n_small_mape_df = df.nsmallest(n, 'MAPE test average(%)')
    return n_small_mape_df['Input'].values.tolist()


def get_average_df(df):
    features_list = df['Input'].unique()
    results_list = []
    for features in features_list:
        target_df = df[df['Input'] == features]
        mape_test_ave = target_df['MAPE test(%)'].mean()
        mape_val_ave = target_df['MAPE val(%)'].mean()
        mape_train_ave = target_df['MAPE train(%)'].mean()
        results = {
            'ML': 'lgb',
            'loss': 'l2(rmse)',
            'Input Num': len(ast.literal_eval(features)),
            'Input': features,
            'MAPE train average(%)': round(mape_train_ave, 5),
            'MAPE val average(%)': round(mape_val_ave, 5),
            'MAPE test average(%)': round(mape_test_ave, 5)
        }
        results_list.append(results)
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('MAPE test average(%)', ignore_index=True)
    return results_df


def get_top_mape_data(df):
    features_list = df['Input'].unique()
    for features in features_list:
        target_df = df[df['Input'] == features]
        # 全サーバーの平均MAPE
        mape_test_ave = target_df['MAPE test(%)'].mean()
        max_mape = target_df['MAPE test(%)'].max()
        print(target_df[target_df['MAPE test(%)'] == max_mape])
    return


if __name__ == "__main__":
    main()
