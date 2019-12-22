# --------------------------------------------------------------------------------------------------------
# 2019/12/10
# src - calculate_features.py
# md
# --------------------------------------------------------------------------------------------------------
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def add_neighbours(n_neighbours=20):
    fp = "/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/processed/"
    tvt = pd.read_csv(fp + 'train_valid_test.csv')
    tvt = tvt[['id', 'x', 'y', 'label']]
    tvt = tvt.set_index('id')
    dist = pdist(tvt[['x', 'y']].values, metric='euclidean')
    dist_matrix = squareform(dist)
    distance_df = pd.DataFrame(dist_matrix, index=tvt.index, columns=tvt.index)

    all_features = pd.DataFrame()
    for id in tvt.index:
        # Calculate the distances from the closest n_neighbours neighbours
        all_distances = distance_df[id].sort_values()
        neighbours_distance = all_distances.iloc[:n_neighbours]
        neighbours_distance = pd.DataFrame(neighbours_distance.iloc[:n_neighbours]).transpose()
        neighbours_distance.columns = [f'd_{i}' for i in range(n_neighbours)]
        # Calculate the labels from the closest n_neighbours neighbours

        neighbours_labels = pd.DataFrame(tvt.loc[all_distances.iloc[:n_neighbours].index, ['label']]).transpose()
        neighbours_labels.columns = [f'l_{i}' for i in range(n_neighbours)]
        neighbours_labels.index = [id]
        feature = pd.concat([neighbours_distance, neighbours_labels], axis=1)
        all_features = pd.concat([all_features, feature])
    all_features = all_features.drop(columns=['d_0', 'l_0'])  # Delete columns otherwise leaking labels !

    # Merge
    print('merging train')
    train = pd.read_csv(fp + 'train_balanced.csv')
    train_plus = pd.merge(train, all_features, left_on='old_id', right_on=all_features.index)
    train_plus.to_csv(fp + 'train_plus.csv')

    print('merging valid')
    valid = pd.read_csv(fp + 'valid.csv')
    valid_plus = pd.merge(valid, all_features, left_on='old_id', right_on=all_features.index)
    valid_plus.to_csv(fp + 'valid_plus.csv')

    print('merging test')
    test = pd.read_csv(fp + 'test.csv')
    test_plus = pd.merge(test, all_features, left_on='old_id', right_on=all_features.index)
    test_plus.to_csv(fp + 'test_plus.csv')


if __name__ == '__main__':
    add_neighbours()
