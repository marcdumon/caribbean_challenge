# --------------------------------------------------------------------------------------------------------
# 2019/02/21
# my_toolbox.py
# md
# --------------------------------------------------------------------------------------------------------

# Todo:
#  - Make a package of all my personal "helper" scripts to import in different projects
#  - Change ''.format() to f''

import math
import os
import random
import time

from datetime import datetime
from typing import Tuple, List, Dict, Collection, Optional
import numpy as np
import cv2 as cv
import sklearn as skl
from sklearn.decomposition import PCA


# from settings import CONFIG


class MyLogTools:
    """
    Collection of function to display or write logs to file of database
    Level:
        - 1 = OK, WARNING, ERROR
        - 2 = WARNING, ERROR
        - 3 = ERROR
        - 4 = INFO
        - 5 = DEBUG
        - 0  = ALL
    Medium:
        - 1   = console
        - 10  = file
        - 100 = database
    110 means logging to csv-file and database

    Implemented functions are:
        -
    """
    level = 0
    medium = 1

    @classmethod
    def log(cls, msg: str, level=level, medium=medium):
        if level == 0:  # Console
            if medium == 1:
                now = datetime.now()
                print('{}\t{}'.format(now, msg))
            if medium == 2:
                pass
            if medium == 3:
                pass
        if level == 5:  # Console
            if medium == 1:
                now = datetime.now()
                print('{}\t{}'.format(now, msg))


class MyImageTools:
    """
    Collection of functions to amnipulate images

    Implemented functions are:
        - def get_image(cls, iname: str, path: str) -> np.array:
        - def save_image(cls, im_array: np.array, path: str, iname: str) -> None:
        - def symlink_image(cls, path_src: str, path_dst: str, iname: str):
        - autocrop(cls, im_array: np.array, square_min_box: bool = False) -> np.array:
        - resize(cls, im_array: np.array, size: int = 32, interpolation=cv.INTER_AREA) -> np.array:
        - minmax(cls, im_array: np.array, per_channel: bool = False) -> np.array:
        - stdize(cls, im_array: np.array, mean: List, std: List) -> np.array:
        - gray(cls, im_array: np.array) -> np.array:

    """
    """ 
    Todo
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images 
    of shape (3 x H x W), where H and W are expected to be at least 224. 
    The images have to be loaded in to a range of [0, 1] and then 
    normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
    You can use the following transform to normalize:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """

    @classmethod
    def get_image(cls, iname: str, path: str) -> np.array:
        """
        Gets image with name 'im_name' located in 'path' directory
            Args:
                - im_name: The filename of the image
                - path: The path where the image exists
            Returns:
                Returns a np.array
        """
        im_array = cv.imread(path + iname, 1)
        if im_array is None:
            MyLogTools.log('ERROR: No image array loaded: {}{}'.format(path, iname))
            raise FileNotFoundError
        # MyLogTools.log('DEBUG: Image loaded: {}'.format(iname), level=5)
        return im_array

    @classmethod
    def save_image(cls, im_array: np.array, path: str, iname: str) -> None:
        succ = cv.imwrite(path + iname, im_array)
        if not succ:  # image could not be written
            MyLogTools.log('ERROR: No image array saved: {}{}'.format(path, iname))
            MyLogTools.log('ERROR: No image array saved. Shape: {}'.format(im_array.shape))
            raise Exception

    @classmethod
    def symlink_image(cls, path_src: str, path_dst: str, iname: str):
        os.symlink(path_src + iname, path_dst + iname)

    @classmethod
    def autocrop(cls, im_array: np.array, inside: bool = False, expand: float = 1.0) -> np.array:
        """
        Automatically crops an image and make it quare.
        Doing autocrop before resizing avoids change in aspect ratio for rectangle images
            Args:
                im_array: The image to be cropped
                inside: If True, the image will be croped with the biggest rectangle inside the eye-circle.
                        A part of the image will be lost, but most lost part is from the background. The image size should be half of the outside box.
                        If False it will cropped with the minimum rectange ouitside the eye-cicle.
                expand: Factor to expand the inside square.
            Returns:
                Returnes the croped square image
        """
        # blured = im_array.filter(ImageFilter.BoxBlur(20))
        blured = cv.blur(im_array, (5, 5))
        # Convert to grayscale
        blured = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
        # Convert to B&W
        _, bw = cv.threshold(blured, 50, 255, cv.THRESH_BINARY)  # Eye circle is white, rest black
        # Get the bounding rectangle (see: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html)
        x, y, w, h = cv.boundingRect(bw)  # outside the eye-circle
        channels = im_array.shape[2]
        aoi = im_array[y:y + h, x:x + w]  # Area of Interest = rectangular raround the eye

        # outside square
        # Create a black square image
        l = max(h, w)
        im_array = np.zeros((l, l, channels), np.uint8)
        # put aoi in black image
        im_array[int((l - h) / 2):int((l + h) / 2), int((l - w) / 2):int((l + w) / 2)] = aoi

        if inside:  # inside square
            diameter = im_array.shape[0]  # the quare cropped image
            l = diameter / math.sqrt(2)  # sqrt(a^2 + a^2)
            l = l * expand  # make the quare bigger by factor expand
            im_array = aoi[int((h - l) / 2):int((h + l) / 2), int((w - l) / 2):int((w + l) / 2)]
        return im_array

    @classmethod
    def resize(cls, im_array: np.array, size: int = 32, interpolation=cv.INTER_AREA) -> np.array:
        """
        Resizes an im_array to size size.
            Args:
                im_array: The im_array to be resised
                size: The size in px to wich the im_array shoul be resized
                interpolation: the im_array resampling algorithm. For comparisson see: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
            Return:
                Returns the resized im_array
        """
        im_array = cv.resize(im_array, (size, size), interpolation=interpolation)
        return im_array

    @classmethod
    def augment(cls, im_array: np.ndarray, aug_type: Optional[str] = None, rnd: bool = True) -> dict:
        """
        Augment the im_array by rotation or flipping. Currently the rotation angle and the flipping axis are random
            Args:
                im_array: The im_array to augment
                aug_type: The type of augmentation. Can be 'rotation', 'flip'
                rnd: If True, then a random augmentation will be selected
            Return:
                 Returns the augmented image
        """

        aug_name = ''
        if rnd:
            aug_type = random.choice(
                ['flip', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate', 'rotate',
                 'rotate', 'rotate', 'rotate'])  # many more rotate becasue otherwise the chance to have duplicates with flips are big for small datasets

        if aug_type == 'flip':
            axis = random.choice(['horizontal', 'vertical', 'both'])
            im_array = cls.flip(im_array=im_array, axis=axis)
            aug_name = 'flip' + '_' + axis
        if aug_type == 'rotate':
            angle = random.randint(0, 360)
            im_array = cls.rotate(im_array=im_array, angle=angle)
            aug_name = 'rotate' + '_' + str(angle)

        return {'aug_name': aug_name, 'im_array': im_array}

    @classmethod
    def flip(cls, im_array: np.array, axis) -> np.ndarray:
        if axis == 'horizontal': im_array = cv.flip(im_array, 0)
        if axis == 'vertical': im_array = cv.flip(im_array, 1)
        if axis == 'both': im_array = cv.flip(im_array, -1)
        return im_array

    @classmethod
    def affine_transform(cls, im_array: np.array,
                         src: np.array = np.float32([[1, 0], [0, 1], [1, 1]]),
                         dst: np.array = np.float32([[1, 0], [0, 1], [1, 1]])):
        rows, cols, ch = im_array.shape
        matrix = cv.getAffineTransform(src, dst)
        im_array = cv.warpAffine(im_array, matrix, (cols, rows))
        return im_array

    @classmethod
    def rotate(cls, im_array: np.array, angle: int = 0) -> np.ndarray:
        rows, cols = im_array.shape[:2]

        M = cv.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
        im_array = cv.warpAffine(im_array, M, (cols, rows))
        return im_array


    @classmethod
    def histogram_eqalization(cls, im_array: np.array) -> np.ndarray:

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # split img in bgr
        b, g, r = cv.split(im_array)

        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)

        im_array = cv.merge((b, g, r))

        return im_array

    @classmethod
    def random_pca(cls, im_array: np.array, n_components: int = 64, reverse=False) -> np.ndarray:
        # See https://shankarmsy.github.io/posts/pca-sklearn.html
        pca = PCA(n_components=n_components, whiten=False, svd_solver='randomized')
        # pca needs 2 dim
        height, width, channels = im_array.shape

        im_array = np.reshape(im_array, (height, width * channels))
        im_array = pca.fit_transform(im_array)

        if reverse:
            im_array = pca.inverse_transform(im_array)
            im_array = np.reshape(im_array, (height, width, channels))
            im_array = im_array.astype(np.uint8)

        return im_array

    @classmethod
    def minmax(cls, im_array: np.array, per_channel: bool = False) -> np.array:
        """
        Scales a numpy image array per channel to [0, 1] and returns that array.
            Args:
                - im_array: numpy array to be min-max scaled per channel
                - per_channel: if True, then the min and max will be calculated per channel. Otherwise min and max will be calculated for the entire image
            Return:
                Returns the min-max-scaled numpy array of the image
        """
        n_chanels = im_array.shape[-1]
        print(n_chanels)
        if per_channel:
            new_array = np.zeros(im_array.shape)
            for i in range(n_chanels):  # RGB
                m, M = im_array[..., i].min(), im_array[..., i].max()
                print(i, m, M)
                new_array[..., i] = (im_array[..., i] - m) / (M - m)
        else:
            new_array = (im_array - im_array.min()) / (im_array.max() - im_array.min())
        return new_array

    @classmethod
    def stdize(cls, im_array: np.array, mean: List, std: List) -> np.array:
        """
        See: https://github.com/tensorpack/tensorpack/issues/789
        Makes the features have 0-mean and unit (1) variance (or std). If the im_array is in BGR (OpenCV) iso RGB then the
        mean en std lists should also have that same order! If they are RGB means or std, swap color channels with
        mean[::-1] and std[::-1]
        Models are specific on the mean and std of the imput they are trained on! If you use a 3th party pretrained model,
        stdize the new im_array with the same mean and std as the pictures used to train the model on.
        Ex: Image Net used to train Resnet has mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

            Args:
                - im_array: the im_array to stdize. The im_array should be scaled to [0, 1]
                - mean: The mean  is a ist with value's for each color channel.
                - str: The standard deviation is a List with vaue's for each color schannel.
            Return:
                 Returns the standardized im_array
        """
        n_chanels = im_array.shape[-1]
        new_array = np.zeros(im_array.shape)
        for i in range(n_chanels):  # RGB
            new_array[..., i] = (im_array[..., i] - mean[i]) / std[i]
        return new_array

    @classmethod
    def gray(cls, im_array: np.array) -> np.array:
        return cv.cvtColor(im_array, cv.COLOR_BGR2GRAY)

    @classmethod
    def sift(cls, im_array: np.array, nfeatures=0, nOctaveLayers: int = 81, edgeThreshold: int = 100, sigma: float = 1.6) -> np.array:
        sift = cv.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=81, edgeThreshold=100, sigma=1.6)
        keypoints, descriptors = sift.detectAndCompute(im_array, None)
        im_array = cv.drawKeypoints(im_array, keypoints, None)
        return im_array

    @classmethod
    def ___calc_channel_mean_std(cls):
        """
        Calculates the per channel mean and std for a dataset.
        :return:
        """
        pass

    @classmethod
    def ___calc_mean_std_dataset(cls):
        # Multiprocess iterate over images
        # def get_mean_and_std(dataset): https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py
        #
        pass


class MyOsTools:
    """
    Collection of functions that do OS file and directory operations like read, write, create, rename and check.

    Implemented functions are:
        - get_filenames(cls, path: Path, ext: str = '') -> List:
        - create_directory(cls, path: Path, name: str) -> Dict:
        - move_files(cls, fnames: List, path_src: Path, path_dst: Path) -> Dict:
        - check_dir_exists(cls, path: Path, name: str) -> Dict:
        - check_files_exist(cls, fnames: List, path: Path) -> Dict:
        - delete_file
    """

    @classmethod
    def check_files_exist(cls, fnames: List, path: str) -> Dict:
        """
        Checks if all the files from the list fnames exists in directory path
            Args:
                fnames: List of filenames to be checkt
                path: Path to directory where the files from fnames should exist

            Return:
                 Returns a dict {success, message, fmissing, fexcess}
        """
        succ = True

        filenames = MyOsTools.get_filenames(path)
        fmissing = list(set(fnames) - set(filenames))
        fexcess = list(set(filenames) - set(fnames))
        print(fexcess)
        print(filenames)
        print(fnames)

        if not fexcess:
            succ = succ and True
            msg = 'OK: All {} \t files exist in directory: {}'.format(len(fnames), path)
            MyLogTools.log(msg)
        else:
            succ = succ and False
            msg = 'ERROR: These {} \tfiles are missing in directory {}: {}{}'.format(len(fmissing), path, fmissing[:30], '...' if len(fmissing) > 30 else '')
            MyLogTools.log(msg)
        return {'success': succ, 'message': msg, 'fmissing': fmissing, 'fexcess': fexcess}

    @classmethod
    def check_dir_exists(cls, path: str) -> Dict:
        """
        Checks if the directory exists

        Args:
            path: Path to the directory to check

        Returns:
            Returns a dict {success, message}
        """
        path_dir = os.fspath(path)  # Path -> PathLike
        succ = os.path.isdir(path_dir)
        if succ:
            msg = 'OK: Directory Exists: {}'.format(path_dir)
            MyLogTools.log(msg)
        else:
            msg = 'WARNING: Directory Doesn\'t Exist: {}'.format(path_dir)
            MyLogTools.log(msg)
        return {'success': succ, 'message': msg}

    @classmethod
    def move_files(cls, fnames: List, path_src: str, path_dst: str) -> Dict:
        """
        Moves all the files in the fnames list from path_src to path_dst. If the file already exists,
        it will be silently overwritten. If the file doesn't exist in the path_src, an error will be trown and
        the list with missing files will be returned

        Args:
            fnames: List of file names.
            path_src: Path to source directory
            path_dst: Path to destination directory

        Return:
            Returns a dict {succ, msg, missing_files}
        """
        succ = True
        errors = []
        msgs = []
        tot = len(fnames)
        for i, f in enumerate(fnames):
            try:
                os.rename(path_src + f, path_dst + f)
                succ = succ and True
                msg = 'OK: File {}/{} moved to : {}/{}'.format(i, tot, path_dst, f)
                MyLogTools.log(msg)
            except FileNotFoundError:
                succ = succ and False
                msg = 'ERROR: File {}/{} not found: {}'.format(i, tot, f)
                MyLogTools.log(msg)
                msgs.append(msg)
                errors.append(f)
        return {'success': succ, 'messages': msgs, 'errors': errors}

    @classmethod
    def get_directories(cls, path: str) -> List:
        """
        Reads the sub-directories located in path.
        Args:
            path: Path to the directory where the sub-directories are located

        Returns:
            Returns a alphabetically sorted list of sub-directories located in path
        """
        dir_lst = next(os.walk(path))[1]
        return dir_lst

    @classmethod
    def get_filenames(cls, path: str, ext: str = '') -> List:
        """
        Reads the filenames all the filenames in directory located in path or if ext is supplied,
        reads only the filenames with that extension.

        Args:
            path: Path to the directory where the files are located
            ext: Extension to filter the returned filenames

        Returns:
            Returns a (alphabetically) sorted list of filenames
        """
        directory = os.scandir(path)
        fnames = [f.name for f in directory if f.is_file()]
        if ext:
            fnames = [f for f in fnames if f.split('.')[1] == ext]
        directory.close()
        fnames.sort()
        return fnames

    @classmethod
    def create_directory(cls, path: str) -> Dict:
        """
        Creates a new directory located in path. If the directory already exists, returns success flag = False
        If the directory already exists, a warning will be thrown and the list with existing directory will be returned

        Args:
            path: Path to location where the new directory should be created

        Returns:
            Returns a dict {success, message}

        """
        msgs = []
        path_dir = os.fspath(path)  # Path -> PathLike
        try:
            os.mkdir(path=path_dir)
            succ = True
            msg = 'OK: Directory Created: {}'.format(path_dir)
            MyLogTools.log(msg)
        except FileExistsError:
            succ = False
            msg = 'WARNING: Directory Exists: {}'.format(path_dir)
            MyLogTools.log(msg)
            msgs.append(msg)
        except FileNotFoundError:
            succ = False
            msg = 'ERROR: Path doesn\'t exists: {}'.format(path_dir)
            MyLogTools.log(msg)
            msgs.append(msg)
        return {'success': succ, 'message': msgs}

    @classmethod
    def delete_files(cls, fnames: List, path: str) -> Dict:
        succ = True
        errors = []
        msgs = []
        tot = len(fnames)
        for i, f in enumerate(fnames):
            try:
                os.remove(path + f)
                msg = f'OK: File {i}/{tot} moved to : {path}/{f}'
                MyLogTools.log(msg)
            except FileNotFoundError:
                succ = succ and False
                msg = f'ERROR: File {i}/{tot} not found: {f}'
                MyLogTools.log(msg)
                msgs.append(msg)
                errors.append(f)
        return {'success': succ, 'messages': msgs, 'errors': errors}

    # ----------------------------------------------------------------------------------------------------------------------
    @classmethod
    def ___delete_directory(cls):
        pass

    @classmethod
    def ___rename_directory(cls):
        pass

    @classmethod
    def ___rename_files(cls) -> Tuple[bool, str]:
        """
        Renames all the files in the old_names list to names in the new_names list.

        Return:
            Returns a tuple (succ, msg)
        """
        succ = False
        msg = ''
        return succ, msg
    # ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    xsdfsdf = 'yyy'
    # # x = MyOsTools.get_dir_filesnames(Path('../data/0_original/train'), 'jpeg')
    # # x = MyOsTools.create_directory(path=Path('../data/0_original/'), name='test2')
    # MyLogTools.log('xxx')
    # print(x)
    # print(len(x))
