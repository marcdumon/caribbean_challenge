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
from my_tools.my_toolbox import MyImageTools as mit
import matplotlib.pyplot as plt


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
        # {'country': 'colombia', 'place': 'borde_rural'},
        # {'country': 'colombia', 'place': 'borde_soacha'},
        # {'country': 'guatemala', 'place': 'mixco_1_and_ebenezer'},
        {'country': 'guatemala', 'place': 'mixco_3'},
        # {'country': 'st_lucia', 'place': 'castries'},
        # {'country': 'st_lucia', 'place': 'dennery'},
        # {'country': 'st_lucia', 'place': 'gros_islet'}
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
            train_geo_df.to_crs(crs=tiff_crs, inplace=True)
            # Convert the test geometry crs, if it exists, to the tiff crs
            if not test_geo_df.empty: test_geo_df.to_crs(crs=tiff_crs, inplace=True)  # Neither the Castries and Gros Islet have a "test-" GeoJSON file.

            # Todo: combine train_valid and test loops. They are almost the same.
            # Loop through each train roof
            print('Cropping train roofs')
            for _, roof in train_geo_df.iterrows():
                break
                roof_id = roof['id']
                print(roof_id, end=' ', flush=True)
                roof_label = roof['roof_material']
                roof_label = roof['roof_material']
                roof_geometry = roof['geometry']
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

                # check if csv-files exists, if not create it
                if not os.path.isfile(fp_processed + 'train_valid_test.csv'):
                    fieldnames = ['id', 'country', 'place', 'verified', 'area', 'complexity', 'x', 'y', 'label', 'test']
                    with open(fp_processed + 'train_valid_test.csv', 'w') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writeheader()

                # append roof features to csv file
                row = [roof_id, country, place, roof_verified, area, complexity, x, y, roof_label, False]
                with open(fp_processed + 'train_valid_test.csv', 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                # delete the aux.xml file
                os.remove(f'{fp_processed}train_valid_test/{roof_id}.png.aux.xml')
            print()

            # Loop through each test roof
            print('Cropping test roofs')
            for _, roof in test_geo_df.iterrows():
                roof_id = roof['id']
                print(roof_id, end=' ', flush=True)
                #roof_label = roof['roof_material']
                #roof_label = roof['roof_material']
                roof_geometry = roof['geometry']
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

                # check if csv-files exists, if not create it
                if not os.path.isfile(fp_processed + 'train_valid_test.csv'):
                    fieldnames = ['id', 'country', 'place', 'verified', 'area', 'complexity', 'x', 'y', 'label', 'test']
                    with open(fp_processed + 'train_valid_test.csv', 'w') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writeheader()

                # append roof data to csv file
                row = [roof_id, country, place, False, area, complexity, x, y, '', True]
                with open(fp_processed + 'train_valid_test.csv', 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(row)
                # delete the aux.xml file
                os.remove(f'{fp_processed}train_valid_test/{roof_id}.png.aux.xml')


def split_train_valid(valid_pct=.25, fp_processed=''):
    train_valid = pd.read_csv(fp + 'processed/train_valid_balanced.csv')
    # Split train/valid. Valid cannot contain Castries and Gros Islet areas because
    # neither the Castries and Gros Islet areas have a "test-" GeoJSON file. (valid should reflect the test distribution)
    # This is because those ground truth labels are from unverified predictions and so are not tested against.
    valid_roofs = train_valid.query('place not in ["castries", "gros_islet"]')
    # The train_valid_balanced contains multiple images from the same roofs. Use id_prev to select and not id!!!
    valid_id_prevs = valid_roofs['id_prev'].unique()  # numpy.ndarray
    valid_id_prevs = pd.Series(valid_id_prevs)

    all_id_prevs = train_valid['id_prev'].unique()  # numpy.ndarray
    all_id_prevs = pd.Series(all_id_prevs)

    nr_valid = int(len(valid_id_prevs) * valid_pct)
    valid_id_prevs = valid_id_prevs.sample(nr_valid)
    train_id_prevs = all_id_prevs[~all_id_prevs.isin(valid_id_prevs)]
    train = train_valid[train_valid['id_prev'].isin(train_id_prevs)]
    valid = train_valid[train_valid['id_prev'].isin(valid_id_prevs)]

    train.to_csv(fp_processed + 'train.csv')
    valid.to_csv(fp_processed + 'valid.csv')


def balance_train_valid(samples_per_class=100, fp_processed=''):
    train_valid = pd.read_csv(fp_processed + 'train_valid_test.csv')
    train_valid_balanced = pd.DataFrame(columns=train_valid.columns)
    for i in train_valid['label'].unique():
        trn_vld = train_valid[train_valid['label'] == i].sample(samples_per_class, replace=True)
        trn_vld.reset_index(inplace=True)
        trn_vld['id_prev'] = trn_vld['id']
        trn_vld['id'] = trn_vld['id'] + '_' + trn_vld.index.map(str)  # Otherwise id is not unique
        train_valid_balanced = pd.concat([train_valid_balanced, trn_vld], sort=False)
    train_valid_balanced.to_csv(fp_processed + 'train_valid_balanced.csv')
    return train_valid_balanced


def augment_and_save_roofs(train_balanced, h_w=256, fp_processed=''):
    for i, row in train_balanced.iterrows():
        im = cv.imread(fp_processed + f'train_valid/{row["id_prev"]}.png')
        # Random affine transform
        a = 1 - np.random.rand() / 5
        b = np.random.rand() / 2
        c = np.random.rand() / 2
        d = 1 - np.random.rand() / 5
        src = np.float32([[1, 0], [0, 1], [1, 1]])
        dst = np.float32([[a, b], [c, d], [1, 1]])
        im = mit.affine_transform(im, src, dst)
        # Random rotate
        angle = np.random.randint(0, 359)
        im = mit.rotate(im, angle)
        ax = np.random.choice(['horizontal', 'vertical', 'both', ''])
        im = mit.flip(im, axis=ax)
        # # Histogran equalizatrion
        im = mit.histogram_eqalization(im)
        # Autocrop
        im = mit.autocrop(im)
        try:
            im = mit.resize(im, h_w)
        except:
            print('-' * 150)
            print(row)
            print(a, b, c, d)
            im = cv.imread(fp_processed + f'train_valid/{row["id_prev"]}.png')
            plt.imshow(im)
            plt.show()
            im = mit.affine_transform(im, src, dst)
            plt.imshow(im)
            plt.show()
            print('-' * 150)

        cv.imwrite(fp_processed + f'train_valid_augment/{row["id"]}.png', im)
        print(f'{i}-{row["id_prev"]}', end=' ', flush=True)


# Todo: improve spagetti code. Preprocess valid in preprocess_image and train in augment. No test preprocess implemented
def preprocess_valid_images(fp_processed=''):
    valid = pd.read_csv(fp_processed + 'valid.csv')
    for i, row in valid.iterrows():
        im = cv.imread(fp_processed + f'train_valid/{row["id"]}.png')
        # im = mit.histogram_eqalization(im)
        cv.imwrite(fp_processed + f'valid/{row["id"]}.png', im)
        print(f'{i}-{row["id"]}', end=' ', flush=True)


if __name__ == '__main__':
    # Define the filepaths
    fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
    fp_i = f'{fp}interim/'
    fp_p = f'{fp}processed_temp/'

    # Create the dataset
    crop_and_save_roofs(fp_i, fp_p)

    # Split train / valid
    # split_train_valid(0.99999, fp_processed=fp_p)

    # Balance the dataset
    # tv_b = balance_train_valid(10000, fp_processed=fp_p)
    # print(train_valid_balanced)

    # Create augmented dataset
    # augment_and_save_roofs(tv_b, 1024, fp_processed=fp_p)

    # Split train/valid
    # split_train_valid(.25, fp_processed=fp_p)
