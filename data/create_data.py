# --------------------------------------------------------------------------------------------------------
# 2019/11/16
# src - create_data.py
# md
# --------------------------------------------------------------------------------------------------------
import csv
import os

import cv2 as cv
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterstats
from pandas import CategoricalDtype
from rasterio.mask import mask
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
from skimage.color import rgb2gray
from skimage.draw import polygon_perimeter

pd.options.display.max_columns = 15
pd.options.display.width = 0
pd.set_option('expand_frame_repr', True)
pd.options.mode.chained_assignment = None


def crop_and_save_roofs(fp_interim='', fp_processed='', size=500):
    """
    This method does following processes on the tiff-files located in fp_interim to generate the roofs datasets in fp_processed:
        - Read the tiff-file located in the fp_interim path for the country and place
        - Read the GeoJSON files containing the geometries for different roofs
        - Convert the geometries to the same crs as the tiff file
        - Creates a box around the centroid of the geometry
        - Crop and save that box in train_valid_test_area
        - Load the box and mask the geometry
        - Draw the geometry on the box and save it in train_valid_test_coontours
        - Crop and safe the roof in train_valid_test_roofs
        - Create features
        - Add entry to the train_valid_test.csv

    params:
        :param fp_interim: the path to interin data
        :param fp_processed: the path to processed data
        :param size: the size in m2 of the area to crop the roof
    """
    locations = [
        {'country': 'colombia', 'place': 'borde_rural'},
        {'country': 'colombia', 'place': 'borde_soacha'},
        {'country': 'guatemala', 'place': 'mixco_1_and_ebenezer'},
        {'country': 'guatemala', 'place': 'mixco_3'},
        {'country': 'st_lucia', 'place': 'castries'},
        {'country': 'st_lucia', 'place': 'dennery'},
        {'country': 'st_lucia', 'place': 'gros_islet'}
    ]

    for loc in locations:
        country = loc['country']
        place = loc['place']

        fp_interim_place = f'{fp_interim}stac/{country}/{place}/'
        print(f'{"-" * 120}\nStart processing: {country}\t\t{place}\n{"-" * 120}')

        # Load train and test GEOjson in a GeoDataFrame. Columns: id, roof_material, verified, geometry
        train_geo_df = gpd.read_file(f'{fp_interim_place}train-{place}.geojson')
        test_geo_df = gpd.GeoDataFrame()  # Neither the Castries and Gros Islet have a "test-" GeoJSON file.
        if place not in ['castries', 'gros_islet']:
            test_geo_df = gpd.read_file(f'{fp_interim_place}test-{place}.geojson')

        # Open the Tiff file
        tiff_file = rasterio.open(f'{fp_interim_place}{place}_ortho-cog.tif', 'r')
        tiff_crs = tiff_file.crs.data
        # Convert the train geometry crs to the tiff crs
        train_geo_df.to_crs(crs=tiff_crs, inplace=True)  # Todo: PyProj FutureWarning: '+init=<authority>:<code>' syntax is deprecated.
        # Convert the test geometry crs, if it exists, to the tiff crs
        if not test_geo_df.empty: test_geo_df.to_crs(crs=tiff_crs, inplace=True)  # Neither the Castries and Gros Islet have a "test-" GeoJSON file.
        # Prepare the profile for the cropped image
        profile = tiff_file.profile

        for is_test, geo_df in zip([False, True], [train_geo_df, test_geo_df]):
            # check if csv-files exists, if not create it
            if not os.path.isfile(fp_processed + 'train_valid_test.csv'):
                fieldnames = ['id', 'country', 'place', 'verified', 'area', 'complexity', 'x', 'y'] + \
                             ['z_min', 'z_max', 'z_median', 'z_count', 'z_majority', 'z_minority', 'z_unique', 'z_range', 'z_sum'] + \
                             ['label', 'test']
                with open(fp_processed + 'train_valid_test.csv', 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

            # Loop through each train/test roof
            print(f'Cropping roofs')
            for _, roof in geo_df.iterrows():
                roof_id = roof['id']
                roof_geometry = roof['geometry']
                if is_test:
                    roof_label = None
                    roof_verified = False
                else:
                    roof_label = roof['roof_material']
                    roof_verified = roof['verified']
                print(roof_id, end=' ', flush=True)

                # First create a box of size m2 around the center of the roof
                centroid = roof_geometry.centroid
                box = centroid.buffer(np.sqrt(size) / 2, cap_style=3)  # 3 -> box
                # Crop the tiff image
                box, transform = rasterio.mask.mask(tiff_file, [box], crop=True)
                profile['count'] = 3  # We'll save the box in RGB and not RGB+Alpha
                profile['driver'] = 'PNG'
                profile['width'] = box.shape[1]
                profile['height'] = box.shape[2]
                profile['transform'] = transform
                # Remove profile keys that doen't exist in PNG file
                [profile.pop(key, None) for key in ['blockxsize', 'blockysize', 'tiled', 'compress', 'interleave']]
                # Save the box to png file
                box = box[:3, ::]  # destroy the alpha channel for saving: 4*520*520 to 3*520*520

                with rasterio.open(f'{fp_processed}train_valid_test_areas/{roof_id}.png', 'w', **profile) as png_file:
                    png_file.write(box)

                # Second mask the roof geometry from the box => image size is same as box
                box_file = rasterio.open(f'{fp_processed}train_valid_test_areas/{roof_id}.png', 'r')
                roof, _ = rasterio.mask.mask(box_file, [roof_geometry], crop=False)
                roof = roof.transpose(1, 2, 0)
                # Draw the contours
                roof_gray = rgb2gray(roof)  # 520*520
                contours = measure.find_contours(roof_gray, .01)
                #   Assuming the roof contour is the fist contour, others are just artifacts

                if len(contours) == 0:  # The geometry lies completely outside the box. Draw contour around the box
                    h = np.arange(1, box.shape[1])
                    w = np.arange(1, box.shape[1])
                    s = min(box.shape[1:])  # the smallest x,y coordinate of box
                    h_coord = np.concatenate([h, [3] * s, h, [max(h) - 3] * s])
                    w_coord = np.concatenate([[3] * s, w, [max(w) - 3] * s, w])
                    contour = np.array([[x, y] for x, y in zip(h_coord, w_coord)])

                else:
                    contour = contours[0]

                #   Draw the contour with line_size in the box
                line_size = [-2, -1, 1, 2]
                line_color = [255, 0, 0]  # Red
                box = box.transpose(1, 2, 0)  # from 4*520*520 to 520*520*3
                for l in line_size:
                    rr, cc = polygon_perimeter(contour[:, 0] + l, contour[:, 1] + l, shape=box.shape, clip=False)
                    box[rr, cc, :3] = [line_color]

                # Safe the box with contour
                box = box.transpose(2, 0, 1)  # from 520*520*3 to 3*520*520
                with rasterio.open(f'{fp_processed}train_valid_test_contours/{roof_id}.png', 'w', **profile) as png_file:
                    png_file.write(box)

                # Also save the cropped roof should we need it
                roof, _ = rasterio.mask.mask(box_file, [roof_geometry], crop=True)
                with rasterio.open(f'{fp_processed}train_valid_test_roofs/{roof_id}.png', 'w', **profile) as png_file:
                    png_file.write(roof)

                # delete the aux.xml files
                os.remove(f'{fp_processed}train_valid_test_areas/{roof_id}.png.aux.xml')  # Only delete after rasterio.open !!!
                os.remove(f'{fp_processed}train_valid_test_contours/{roof_id}.png.aux.xml')
                os.remove(f'{fp_processed}train_valid_test_roofs/{roof_id}.png.aux.xml')

                # create features
                area = roof_geometry.area
                complexity = 0
                if type(roof_geometry) == Polygon:
                    complexity = len(roof_geometry.exterior.coords.xy[0]) - 1  # complexity of the shape
                elif type(roof_geometry) == MultiPolygon:  # take the complexity of the biggest shape
                    max_area = 0
                    i = 0
                    for l in range(len(roof_geometry)):
                        max_area = max(max_area, roof_geometry[l].area)
                        if max_area == roof_geometry[l].area: i = l
                    complexity = len(roof_geometry[i].exterior.coords.xy[0]) - 1
                h = roof_geometry.centroid.x
                y = roof_geometry.centroid.y
                # Add the zonal_stats for the roof
                stats = ['min', 'max', 'median', 'count', 'majority', 'minority', 'unique', 'range', 'sum']
                zonal_stats = rasterstats.zonal_stats(roof_geometry, f'{fp}interim/stac/{country}/{place}/{place}_ortho-cog.tif', stats=stats, nodata=-999)
                # append roof features to csv file
                row = [roof_id, country, place, roof_verified, area, complexity, h, y] + [zonal_stats[0][k] for k in zonal_stats[0]] + [roof_label, is_test]
                with open(fp_processed + 'train_valid_test.csv', 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)

            print()


def cleanup_train_valid_test(fp_processed):
    tvt = pd.read_csv(fp_processed + 'train_valid_test.csv')
    # Remove not verified healthy_metal
    tvt = tvt[~((tvt['label'] == 'healthy_metal') & (tvt['verified'] == False))]
    # Remove healthy_metal area < 2.5,
    tvt = tvt[~((tvt['area'] < 2.5) & (tvt['label'] == 'healthy_metal'))]
    tvt.to_csv(fp_processed + 'train_valid_test_clean.csv')


def split_balance_dataset(fp_processed, samples_per_label=5000, valid_pct=.3):
    """
    Splits train_valid_test dataset into train_valid and test.
    Creates a valid dataset with the same distribution of labels as the full dataset and as many unverified roofs as possible,
    becasue it's better to have the verified roofs in the training dataset.
    The test dataset doesn't contain roofs from the places Castries and Gros Islet, therefore the valid dataset also exclude these places.

    :param fp_processed:
    :param samples_per_label:
    :param valid_pct:
    :return:
    """
    tvt = pd.read_csv(fp_processed + 'train_valid_test.csv')
    # Split train, valid, test
    tv = tvt[tvt['test'] == False]
    labels = set(tv['label'])
    test = tvt[tvt['test'] == True]
    #   The valid dataset needs to have the same label distribution as the full dataset
    #   Better put as many unverified roofs as possible in valid and train with verified roofs
    planned_distribution = tv['label'].value_counts(normalize=False)
    planned_distribution = (round(planned_distribution * valid_pct)).astype(int)
    valid = pd.DataFrame(columns=tv.columns)
    tv_unverified = tv[tv['verified'] == False]
    # Valid should include only places that are also in Test
    tv_unverified = tv[tv['place'].isin(set(test['place']))]
    for label in labels:
        count = len(tv_unverified[tv_unverified['label'] == label])
        if planned_distribution[label] <= count:  # we have enough unverified roofs
            v_label = tv_unverified[tv_unverified['label'] == label].sample(planned_distribution[label])
            valid = pd.concat([valid, v_label])
        else:  # we don't have enough unverified roofs => also need verified roofs
            v_label_unverified = tv_unverified[tv_unverified['label'] == label]
            missing = planned_distribution[label] - len(v_label_unverified)
            v_label_verified = tv[(tv['label'] == label) &
                                  (tv['verified'] == True) &
                                  tv['place'].isin(set(test['place']))].sample(missing)
            valid = pd.concat([valid, v_label_unverified, v_label_verified])
    #   Delete the valids from the train
    train = tv[~(tv['id'].isin(valid['id']))]

    # Balance train
    train_balanced = pd.DataFrame(columns=train.columns)
    for label in labels:
        train_label = train[train['label'] == label]
        if len(train_label) >= samples_per_label:
            train_label = train_label.sample(samples_per_label, replace=False)
            train_balanced = pd.concat([train_balanced, train_label])
        else:  # not enough samples, need oversampling
            train_label = train_label.sample(samples_per_label, replace=True)
            train_label.sort_values('id', inplace=True)
            train_balanced = pd.concat([train_balanced, train_label])

    # Reset_indexes
    train_balanced.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Save
    train_balanced.to_csv(fp_processed + 'train.csv')
    valid.to_csv(fp_processed + 'valid.csv')
    test.to_csv(fp_processed + 'test.csv')


def augment_dataset(fp_processed, image_size=512):
    flips = ['non', 'ver', 'hor', 'bot']
    for ds in ['train', 'valid', 'test']:
        print(f'\nStart augmenting {ds}')
        dataset = pd.read_csv(fp_processed + f'{ds}.csv', index_col=0)
        for i, row in dataset.iterrows():
            img = cv.imread(fp_processed + f'train_valid_test_contours/{row["id"]}.png')
            flip = flips[np.random.randint(4)]
            rot = np.random.randint(360)
            if flip == 'non': pass
            if flip == 'ver': img = cv.flip(img, 0)
            if flip == 'hor': img = cv.flip(img, 1)
            if flip == 'bot': img = cv.flip(img, -1)
            img = ndimage.rotate(img, rot, reshape=False, mode='wrap')
            img = cv.resize(img, (image_size, image_size))
            dataset.loc[i, 'id_aug'] = f'{row["id"]}_{flip}_{rot}'

            cv.imwrite(fp_processed + f'train_valid_test_augmented/{row["id"]}_{flip}_{rot}.png', img)
            print(row['id'], end=' ', flush=True)
        dataset.to_csv(fp_processed + f'{ds}.csv')


def add_neighbours(fp_processed, n_neighbours=20):
    tvt = pd.read_csv(fp_processed + 'train_valid_test.csv')
    tvt = tvt[['id', 'x', 'y', 'label']]
    tvt = tvt.set_index('id')
    dist = pdist(tvt[['x', 'y']].values, metric='euclidean')
    dist_matrix = squareform(dist)
    distance_df = pd.DataFrame(dist_matrix, index=tvt.index, columns=tvt.index)

    all_features = pd.DataFrame()
    for tvt_id in tvt.index:
        # Calculate the distances from the closest n_neighbours neighbours
        all_distances = distance_df[tvt_id].sort_values()
        neighbours_distance = all_distances.iloc[:n_neighbours + 1]
        neighbours_distance = pd.DataFrame(neighbours_distance.iloc[:n_neighbours + 1]).transpose()
        neighbours_distance.columns = [f'd_{i}' for i in range(n_neighbours + 1)]
        # Calculate the labels from the closest n_neighbours neighbours
        neighbours_labels = pd.DataFrame(tvt.loc[all_distances.iloc[:n_neighbours + 1].index, ['label']]).transpose()
        neighbours_labels.columns = [f'l_{i}' for i in range(n_neighbours + 1)]
        neighbours_labels.index = [tvt_id]
        feature = pd.concat([neighbours_distance, neighbours_labels], axis=1)
        all_features = pd.concat([all_features, feature])
    all_features = all_features.drop(columns=['d_0', 'l_0'])  # Delete columns otherwise leaking labels !

    # Merge
    print('merging train')
    train = pd.read_csv(fp_processed + 'train.csv', index_col=0)
    train_plus = pd.merge(train, all_features, left_on='id', right_on=all_features.index)
    train_plus.to_csv(fp_processed + 'train.csv')

    print('merging valid')
    valid = pd.read_csv(fp_processed + 'valid.csv', index_col=0)
    valid_plus = pd.merge(valid, all_features, left_on='id', right_on=all_features.index)
    valid_plus.to_csv(fp_processed + 'valid.csv')

    print('merging test')
    test = pd.read_csv(fp_processed + 'test.csv', index_col=0)
    test_plus = pd.merge(test, all_features, left_on='id', right_on=all_features.index)
    test_plus.to_csv(fp_processed + 'test.csv')


def preprocess_features(fp_processed, only_label=True):
    # Load and merge the datasets
    train = pd.read_csv(fp_processed + 'train.csv', index_col=0)
    valid = pd.read_csv(fp_processed + 'valid.csv', index_col=0)
    test = pd.read_csv(fp_processed + 'test.csv', index_col=0)
    #   For easier splitting afterwards
    train['dataset'] = 'train'
    valid['dataset'] = 'valid'
    test['dataset'] = 'test'
    tvt = pd.concat([train, valid, test])

    labels = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']
    countries = ['colombia', 'guatemala', 'st_lucia']
    places = ['borde_rural', 'borde_soacha', 'castries', 'dennery', 'gros_islet', 'mixco_1_and_ebenezer', 'mixco_3']

    countries_cat_type = CategoricalDtype(categories=countries, ordered=True)
    places_cat_type = CategoricalDtype(categories=places, ordered=True)
    labels_cat_type = CategoricalDtype(categories=labels + ['unknown'], ordered=True)  # +['unknown] for the nan's in neighbour labels

    # Encode labels
    tvt.loc[:, 'label'] = tvt.loc[:, 'label'].astype(labels_cat_type).cat.codes

    if not only_label:
        # Encode categories
        #   First handle nan, otherwise cat.code for nan is -1, resulting in error in ebedding (index out of range: -1)
        tvt = tvt.fillna('unknown')

        tvt.loc[:, 'country'] = tvt.loc[:, 'country'].astype(str).astype(countries_cat_type).cat.codes
        tvt.loc[:, 'place'] = tvt.loc[:, 'place'].astype(places_cat_type).cat.codes
        tvt.loc[:, 'verified'] = tvt.loc[:, 'verified'].astype(int)
        for i in range(1, 21):
            tvt.loc[:, f'l_{i}'] = tvt.loc[:, f'l_{i}'].astype(labels_cat_type).cat.codes

        # Normalize continuous features
        continuous_cols = ['area', 'complexity', 'z_min', 'z_max', 'z_median', 'z_count', 'z_majority', 'z_minority',
                           'z_unique', 'z_range', 'z_sum']
        for col in continuous_cols:
            mu = tvt[col].mean()
            sigma = tvt[col].std()
            tvt.loc[:, col] = (tvt[col] - mu) / sigma

        # Normalize distances
        mu = tvt.loc[:, 'd_1':'d_19'].values.mean()
        sigma = tvt.loc[:, 'd_1':'d_19'].values.std()
        for i in range(1, 21):
            tvt.loc[:, f'd_{i}'] = (tvt[f'd_{i}'] - mu) / sigma

    # split and save
    train = tvt[tvt['dataset'] == 'train']
    valid = tvt[tvt['dataset'] == 'valid']
    test = tvt[tvt['dataset'] == 'test']
    train.to_csv(fp_processed + 'train_.csv')
    valid.to_csv(fp_processed + 'valid.csv')
    test.to_csv(fp_processed + 'test.csv')


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Define the filepaths
    fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
    fp_i = f'{fp}interim/'
    fp_p = f'{fp}processed/'

    # Create the dataset
    # crop_and_save_roofs(fp_i, fp_p, size=1200)

    # Cleanup
    # cleanup_train_valid_test(fp_p)

    # Split and balance
    # split_balance_dataset(fp_p, 10000, valid_pct=.30)

    # Augmentation
    # augment_dataset(fp_p, image_size=512)

    # Additional Features
    # add_neighbours(fp_p, n_neighbours=20)

    # Pre-Process features
    # preprocess_features(fp_p, only_label=True)
