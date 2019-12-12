# --------------------------------------------------------------------------------------------------------
# 2019/11/16
# src - create_dataset.py
# md
# --------------------------------------------------------------------------------------------------------

"""
This script creates all the necessary train, valid and test files in /media/md/Datasets/drivendata_open_aI_caribbean_challenge/processed
    - reads the interim tiff files from the interim directory
    - crops the roofs according to the geometry data from the .geojson files from interim directory
    - saves the roofs images in the train or test directory
    - creates the train_valid.csv and test.csv files
    - clean-up
"""
import os
import csv
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split

from my_tools.my_toolbox import MyImageTools as mit
import matplotlib.pyplot as plt
import albumentations as albu


def crop_and_save_roofs(fp_interim='', fp_processed=''):
    """
    This method does following processes on the tiff-files located in fp_interim to generate the roofs datasets in fp_processed:
        - Read the tiff-file located in the fp_interim path for the country and place
        - Read the GeoJSON files containing the geometries for different roofs
        - Convert the geometries to the same crs as the tiff file
        - Crop and safe the roofs
        - Create features
        - Add entry to the train_valid_test.csv

    params:
        :param fp_interim: the path to interin data
        :param fp_processed: the path to processed data
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
        with rasterio.open(f'{fp_interim_place}{place}_ortho-cog.tif', 'r') as tiff_file:
            # Get the profile from the tiff file
            profile = tiff_file.profile
            # Prepare the profile for the cropped image
            profile['driver'] = 'PNG'
            # Get the crs from the tiff file
            tiff_crs = tiff_file.crs.data
            # Convert the train geometry crs to the tiff crs
            train_geo_df.to_crs(crs=tiff_crs,
                                inplace=True)  # Todo: FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method.
            # Convert the test geometry crs, if it exists, to the tiff crs
            if not test_geo_df.empty: test_geo_df.to_crs(crs=tiff_crs, inplace=True)  # Neither the Castries and Gros Islet have a "test-" GeoJSON file.

            for is_test, geo_df in zip([False, True], [train_geo_df, test_geo_df]):
                # check if csv-files exists, if not create it
                if not os.path.isfile(fp_processed + 'train_valid_test.csv'):
                    fieldnames = ['id', 'country', 'place', 'verified', 'area', 'complexity', 'x', 'y', 'label', 'test']
                    with open(fp_processed + 'train_valid_test.csv', 'w') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writeheader()

                # Loop through each train/test roof
                print(f'Cropping roofs')
                for _, roof in geo_df.iterrows():
                    roof_id = roof['id']
                    print(roof_id, end=' ', flush=True)
                    roof_geometry = roof['geometry']
                    if is_test:
                        roof_label = None
                        roof_verified = False
                    else:
                        roof_label = roof['roof_material']
                        roof_verified = roof['verified']

                    # Crop the tiff image
                    roof, _ = rasterio.mask.mask(tiff_file, [roof_geometry], crop=True)
                    profile['width'] = roof.shape[1]
                    profile['height'] = roof.shape[2]

                    # Save the roof to png file
                    with rasterio.open(f'{fp_processed}train_valid_test/{roof_id}.png', 'w', **profile) as png_file:
                        png_file.write(roof)

                    # create features
                    area = roof_geometry.area
                    if type(roof_geometry) == Polygon:
                        complexity = len(roof_geometry.exterior.coords.xy[0]) - 1  # complexity of the shape
                    elif type(roof_geometry) == MultiPolygon:  # take the complexity of the biggest shape
                        max_area = 0
                        i = 0
                        for l in range(len(roof_geometry)):
                            max_area = max(max_area, roof_geometry[l].area)
                            if max_area == roof_geometry[l].area: i = l
                        complexity = len(roof_geometry[i].exterior.coords.xy[0]) - 1
                    x = roof_geometry.centroid.x
                    y = roof_geometry.centroid.y

                    # append roof features to csv file
                    row = [roof_id, country, place, roof_verified, area, complexity, x, y, roof_label, is_test]
                    with open(fp_processed + 'train_valid_test.csv', 'a') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(row)
                    # delete the aux.xml file
                    os.remove(f'{fp_processed}train_valid_test/{roof_id}.png.aux.xml')
                print()


def cleanup_train_valid_test(fp_processed):
    tvt = pd.read_csv(fp_processed + 'train_valid_test.csv')
    # Remove not verified healthy_metal
    tvt = tvt[~((tvt['label'] == 'healthy_metal') & (tvt['verified'] == False))]
    # Remove area < 2.5, but not for test
    tvt = tvt[(tvt['area'] > 2.5) | (tvt['test']==True)]
    tvt.to_csv(fp_processed + 'train_valid_test_clean.csv')


def split_train_valid_test(valid_pct, fp_processed):
    # Split before augmentation:
    #   - Label distribution from valid should reflect the test set
    #   - Avoids that augmented images from the same base image are in both train and valid set
    tvt = pd.read_csv(fp_processed + 'train_valid_test_clean.csv', index_col=0)
    tv = tvt[tvt['test'] == False]
    valid_nr = int(len(tv) * valid_pct)
    # The test set doesn't contain roofs from Castries and Gros Islet, so exclude them in the valid set
    v = tv.query('place not in ["castries", "gros_islet"]')
    _, valid = train_test_split(v, test_size=valid_nr, stratify=(v.loc[:, ['label', 'place']]))
    valid_ids = valid['id']
    train = tv[~tv['id'].isin(valid_ids)]
    test = tvt[tvt['test'] == True]

    # Add old_id # Todo: spagetti code.... need this column in augmentation() + train_balanced[old_id] is added in balance
    valid.loc[:, 'old_id'] = valid['id'] # Todo: warning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
    test.loc[:, 'old_id'] = test['id']

    train.to_csv(fp_processed + 'train.csv')
    valid.to_csv(fp_processed + 'valid.csv')
    test.to_csv(fp_processed + 'test.csv')


def balance(fp_processed, samples_per_label=5000, only_verified=False):
    train = pd.read_csv(fp_processed + 'train.csv', index_col=0)
    if only_verified:
        train=train[train['verified']==True]
    labels = train['label'].unique()
    trn = pd.DataFrame()
    for label in labels:
        print(label)
        trn_label = train[train['label'] == label]
        if len(trn_label) < samples_per_label:  # not enough samples, make sure that we have all samples, then add oversampele
            print(trn_label)
            tv_label_oversamples = train[train['label'] == label].sample(samples_per_label - len(trn_label), replace=True)
            print(tv_label_oversamples)
            trn_label = pd.concat([trn_label, tv_label_oversamples])
            print(trn_label)
        else:  # we have enough samples, no need to oversample
            trn_label = trn_label.sample(samples_per_label)
            print(trn_label)
        print('-' * 150)
        trn_label = trn_label.sort_values('id').reset_index(drop=True)
        trn = pd.concat([trn, trn_label])
        print('-' * 150)
        print(len(trn))
        print('-' * 150)

    # Keep old id and create new one
    trn = trn.reset_index(drop=True).sort_values('id')
    trn['old_id'] = trn['id']
    trn['id'] = trn['old_id'] + '_' + trn.index.map(str)
    print('-' * 150)
    print(len(trn))
    print('-' * 150)
    if only_verified:
        trn.to_csv(fp_processed + 'train_balanced_verified.csv')
    else:
        trn.to_csv(fp_processed + 'train_balanced.csv')


# Fist split then augment! Otherwise the validation set contains augmented train images !!!!!!!
def augmentations(fp_processed, ):
    def augment(p=.5):
        return albu.Compose([
            albu.Rotate((-180, 180), p=1),
            albu.OneOf(
                [albu.HorizontalFlip(),
                 albu.VerticalFlip()], p=.9),
            albu.OneOf([
                albu.RandomBrightnessContrast((-0.3, 0.2), (-0.3, 0.2), p=1),
                albu.RandomGamma((20, 200), p=.9),
                albu.RGBShift(40, 40, 40, p=.9)
            ], p=.5),
            albu.OneOf([
                #     # albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                albu.GridDistortion(num_steps=5, distort_limit=.2),
                albu.OpticalDistortion(distort_limit=.5, shift_limit=22.5, p=.8),
            ], p=.2),
            albu.ShiftScaleRotate(p=.6),
            # albu.Resize(512, 512, always_apply=True),
        ])

    for ds in ['train_balanced', 'valid', 'test']:
    # for ds in [ 'train_balanced_verified', 'valid', 'test']:
    # for ds in [ 'test']:
        dataset = pd.read_csv(fp_processed + f'{ds}.csv')
        for _, row in dataset.iterrows():
            img = cv.imread(fp_processed + f'train_valid_test/{row["old_id"]}.png')
            aug = augment(p=1)
            i = {'image': img}
            img = aug(**i)['image']
            cv.imwrite(fp_processed + f'train_valid_test_augmented/{row["id"]}.png', img)
            print(row['id'], end=' ', flush=True)


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Define the filepaths
    fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
    fp_i = f'{fp}interim/'
    fp_p = f'{fp}processed/'

    # Create the dataset
    # crop_and_save_roofs(fp_i, fp_p)

    # Cleanup
    # cleanup_train_valid_test(fp_p)

    # Split train / valid
    # split_train_valid_test(0.3, fp_p)

    # Balance
    # balance(fp_p, 10000, False)

    # Augmentation
    # augmentations(fp_p)

