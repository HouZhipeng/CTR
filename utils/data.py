# -*- coding:utf-8 -*-
import os
import numpy as np
from keras.utils import Sequence


def _filename_add_suffix(file_path, suffix):
    """
    For a given file path, find the last '.' symbol and add suffix before it.

    :param file_path: The given file path
    :param suffix: The suffix needed to be added before the last '.' symbol
    :return: The path after add suffix
    """
    path_split = file_path.split('.')
    if len(path_split) == 1:
        return file_path + suffix
    else:
        return '.'.join(path_split[:-1]) + suffix + '.' + path_split[-1]


def fill_nan(input_file_path, sparse_field_cols=(), dense_field_cols=(), sep=',',
             sparse_fill_val=-1, dense_fill_val=0, output_file_path=None):
    """
    Fill the missing value of the given input data file.

    :param input_file_path: The original data file path, which may contain missing values
    :param sparse_field_cols: The column indices of sparse fields. (start from 0)
    :param dense_field_cols: The column indices of dense fields. (start from 0)
    :param sep: The value separator used in input data file.
    :param sparse_fill_val: The value used to fill the missing sparse feature
    :param dense_fill_val: The value used to fill the missing dense feature
    :param output_file_path: After this process, the data will be saved in this path.
                             If is assigned as `None`, the default path will be used.
    :return The output file path will be returned
    """
    # check duplication of sparse and dense column index
    if len(set(sparse_field_cols).intersection(dense_field_cols)) > 0:
        raise ValueError('The sparse and dense indices have common value!')
    # Check input and output file
    if not os.path.exists(input_file_path):
        raise FileNotFoundError('The input data file is not exist.')
    if output_file_path is None:
        output_file_path = _filename_add_suffix(input_file_path, '-fillna')
    if os.path.exists(output_file_path):
        raise ValueError('The output data file (%s) is already exist, please check' % output_file_path)
    # iterate the input data file and fill the nan value
    with open(input_file_path, 'r', encoding='utf-8') as fin:
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                data = line.strip('\n').split(sep)
                # check sparse cols
                for col in sparse_field_cols:
                    if data[col] == '':
                        data[col] = str(sparse_fill_val)
                # check dense cols
                for col in dense_field_cols:
                    if data[col] == '':
                        data[col] = str(dense_fill_val)
                # rewrite the data to output file
                fout.write(sep.join(data) + '\n')
    return output_file_path


def min_max_scalar(input_file_path, dense_field_cols, output_file_path=None, sep=','):
    """
    Do min and max scalar operation on each dense field in the given data file

    :param input_file_path: The input data file. (Whole dataset)
    :param dense_field_cols: The column indices of dense fields
    :param output_file_path: After this process, the data will be saved in this path.
                             If is assigned as `None`, the default path will be used.
    :param sep: The value separator
    :return: The output file path
    """
    # Check input and output file
    if not os.path.exists(input_file_path):
        raise FileNotFoundError('The input data file is not exist.')
    if output_file_path is None:
        output_file_path = _filename_add_suffix(input_file_path, '-minmax')
    if os.path.exists(output_file_path):
        raise ValueError('The output data file (%s) is already exist, please check' % output_file_path)

    # first loop, find the minimum and maximum value of each dense field
    minimum = {col: float('inf') for col in dense_field_cols}
    maximum = {col: float('-inf') for col in dense_field_cols}
    with open(input_file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = line.strip('\n').split(sep)
            for col in dense_field_cols:
                val = float(data[col])
                if val < minimum[col]:
                    minimum[col] = val
                if val > maximum[col]:
                    maximum[col] = val

    # second loop, use min & max scalar on each field
    assert (float('inf') not in minimum.values()) and (float('-inf') not in maximum.values())
    difference = {col: (maximum[col] - minimum[col]) for col in dense_field_cols}
    for col in dense_field_cols:
        if difference[col] == 0.0:
            raise ValueError('All the value in dense col %d are same.' % col)
    with open(input_file_path, 'r', encoding='utf-8') as fin:
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                data = line.strip('\n').split(sep)
                for col in dense_field_cols:
                    data[col] = str((float(data[col]) - minimum[col]) / difference[col])
                fout.write(sep.join(data) + '\n')
    return output_file_path


def normalize():
    # TODO: same as `min_max_scalar` function, using `normalize` process on input data
    pass


def sparse_feat_encode(data_file_path, sparse_field_cols, sep=','):
    """
    This function gives each sparse feature an index. The number of sparse features is very large,
    If one-hot encoding is used, the input dimension will be greatly exploded. And, the one-hot
    encode sparse features will be fed into an embedding layer in most machine learning or deep
    learning models. The process of one-hot encode feature multiply embedding matrix can be seen
    as looking up the non-zero index.

    This function will return a tuple, which contain the sparse feature map dict, the number of indices.
    The content of the dict will be liked as follows:

    ```
    feature_map = {
        sparse_field_col_index_1 (int): {sparse_feature_value1 (str): index1 (int),
                                         sparse_feature_value2 (str): index2 (int),
                                         ...},

        ...,

        sparse_field_col_index_n (int): {sparse_feature_value1 (str): index1 (int),
                                         sparse_feature_value2 (str): index2 (int),
                                         ...},
    }
    ```

    :param data_file_path: the dataset file path, which contain the whole dataset.
    :param sparse_field_cols: iterable object. the column index of sparse field. (Start from 0)
    :param sep: the value separator
    :return: a tuple, containing the sparse feature map dict, the number of indices.
    """

    # check dataset file
    if not os.path.exists(data_file_path):
        raise FileNotFoundError('The data file is not exist.')

    sparse_feat_map = {}
    for col in sparse_field_cols:
        sparse_feat_map[col] = {}

    # go through the data file and initialize the feature map
    with open(data_file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            # read content
            data = line.strip('\n').split(sep)
            for col in sparse_field_cols:
                if data[col] not in sparse_feat_map[col].keys():
                    sparse_feat_map[col][data[col]] = -1

    # indexing sparse features
    sparse_index_cnt = 0
    for col, field_feat_dict in sparse_feat_map.items():
        for feat_value in field_feat_dict.keys():
            sparse_feat_map[col][feat_value] = sparse_index_cnt
            sparse_index_cnt += 1

    return sparse_feat_map, sparse_index_cnt


def subsample(input_file_path, sample_num, output_file_path=None):
    """
    Do sub-sampling operation on the given data file

    :param input_file_path: The input data file path
    :param sample_num: The total number of samples in the output data file
    :param output_file_path: After this process, the test data will be saved in this path.
                             If is assigned as `None`, the default path will be used.
    :return: The output file path
    """

    # Check arguments
    if sample_num < 0:
        raise ValueError('The number of sub-samples cannot be a negative value.')
    if not os.path.exists(input_file_path):
        raise FileNotFoundError('The input data file is not exist.')
    if output_file_path is None:
        output_file_path = _filename_add_suffix(input_file_path, '-sample')
    if os.path.exists(output_file_path):
        raise ValueError('The output data file (%s) is already exist, please check' % output_file_path)
    row_no = 0
    with open(input_file_path, 'r', encoding='utf-8') as fin:
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                if row_no < sample_num:
                    fout.write(line)
                    row_no += 1

    return output_file_path


def train_valid_test_split(input_file_path, total_sample_num, train_ratio, valid_ratio,
                           train_file_name=None, valid_file_name=None, test_file_name=None):
    """

    :param input_file_path: The given data file path
    :param total_sample_num: The number of total samples in the given data file
    :param train_ratio: # train-samples / # total-samples
    :param valid_ratio: # valid-samples / # total-samples
    :param train_file_name: After this process, the train data will be saved in this path.
                            If is assigned as `None`, the default path will be used.
    :param valid_file_name: After this process, the valid data will be saved in this path.
                            If is assigned as `None`, the default path will be used.
    :param test_file_name: After this process, the test data will be saved in this path.
                           If is assigned as `None`, the default path will be used.
    :return: a tuple, containing the (train file path, valid file path, test file path)
    """

    # Check arguments
    if total_sample_num < 1:
        raise ValueError('The number total samples should be larger than 1.')
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError('The value of `train_ratio` (%.4f) is illegal' % train_ratio)
    if valid_ratio <= 0 or train_ratio > 1:
        raise ValueError('The value of `valid_ratio` (%.4f) is illegal' % valid_ratio)
    test_ratio = 1 - train_ratio - valid_ratio
    if test_ratio <= 0 or train_ratio > 1:
        raise ValueError('The value of `test_ratio` (%.4f) is illegal' % test_ratio)
    # check file existing
    if not os.path.exists(input_file_path):
        raise FileNotFoundError('The input data file is not exist.')
    if train_file_name is None:
        train_file_name = _filename_add_suffix(input_file_path, '-train')
    if os.path.exists(train_file_name):
        raise ValueError('The output train data file (%s) is already exist, please check' % train_file_name)
    if valid_file_name is None:
        valid_file_name = _filename_add_suffix(input_file_path, '-valid')
    if os.path.exists(valid_file_name):
        raise ValueError('The output valid data file (%s) is already exist, please check' % valid_file_name)
    if test_file_name is None:
        test_file_name = _filename_add_suffix(input_file_path, '-test')
    if os.path.exists(test_file_name):
        raise ValueError('The output test data file (%s) is already exist, please check' % test_file_name)

    train_num = int(np.ceil(total_sample_num * train_ratio))
    valid_num = int(np.ceil(total_sample_num * valid_ratio))

    row_no = 0
    with open(input_file_path, 'r') as fin:
        with open(train_file_name, 'w', encoding='utf-8') as train_out:
            with open(valid_file_name, 'w', encoding='utf-8')as valid_out:
                with open(test_file_name, 'w', encoding='utf-8') as test_out:
                    for line in fin:
                        if row_no < train_num:
                            train_out.write(line)
                        elif row_no < train_num + valid_num:
                            valid_out.write(line)
                        else:
                            test_out.write(line)
                        row_no += 1
    return train_file_name, valid_file_name, test_file_name


class SparseIdxDenseValSeq(Sequence):

    def __init__(self, data_file_path, target_col, sparse_field_cols=(), sparse_feat_map=None,
                 dense_field_cols=(), sep=',', batch_size=4096):
        # check each params
        if not os.path.exists(data_file_path):
            raise FileExistsError('Cannot find the data file!')
        self.data_file_path = data_file_path
        # if have sparse fields
        if len(sparse_field_cols) != 0 and sparse_feat_map is None:
            raise ValueError('Since sparse field is given, the sparse_feat_map must be given.')
        self.sparse_feat_map = sparse_feat_map
        # check the given columns
        assert target_col not in sparse_field_cols and target_col not in dense_field_cols
        if len(set(sparse_field_cols).intersection(dense_field_cols)) > 0:
            raise ValueError('The sparse and dense indices have common value!')
        self.target_col = target_col
        self.sparse_field_cols = sparse_field_cols
        self.dense_field_cols = dense_field_cols
        self.sep = sep
        self.batch_size = batch_size

        # To speed up the speed of load data. The given data file will be split into many
        # small batch files. The line of each batch file is equal to the given batch size.
        # -----------------------------------------------------------------------------------
        self.batch_file_dir = data_file_path + '-batch'
        if os.path.exists(self.batch_file_dir):
            raise ValueError('The batch directory `', self.batch_file_dir, '` already exist')
        os.makedirs(self.batch_file_dir)

        # generate batch data file
        print('=' * 20)
        row_no = 0
        fout = None
        with open(self.data_file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                # Check whether a new file needs to be created
                if row_no % batch_size == 0:
                    if fout:
                        fout.close()
                    file_idx = int(row_no / batch_size)
                    fout = open(os.path.join(self.batch_file_dir, str(file_idx)), 'w', encoding='utf-8')
                    print('Generate batch file', file_idx)
                # write data
                fout.write(line)
                row_no += 1
            if not fout.closed:
                fout.close()
        print('=' * 20)
        assert int(np.ceil(row_no / batch_size)) == len(os.listdir(self.batch_file_dir))
        self.batch_num = len(os.listdir(self.batch_file_dir))
        # -----------------------------------------------------------------------------------

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        if idx >= self.batch_num or idx < 0:
            raise ValueError('The given idx is out of the range: [', 0, ',', self.batch_num - 1, '].')

        sparse_feat_index = []
        dense_feat_value = []
        target = []

        with open(self.data_file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = line.strip('\n').split(sep=self.sep)
                # find target value
                target.append([int(data[self.target_col])])
                # encode the sparse features and find the dense value
                index = []
                value = []
                for col in self.sparse_field_cols:
                    index.append(self.sparse_feat_map[col][data[col]])
                for col in self.dense_field_cols:
                    value.append(float(data[col]))
                sparse_feat_index.append(index)
                dense_feat_value.append(value)

        return [np.array(sparse_feat_index), np.array(dense_feat_value)], np.array(target)

    def clear(self):
        for file_name in os.listdir(self.batch_file_dir):
            os.remove(os.path.join(self.batch_file_dir, file_name))
        os.removedirs(self.batch_file_dir)
