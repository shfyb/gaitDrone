# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False,
                dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    sinfo = img_groups[0]
    img_paths = img_groups[1]

    to_pickle = []
    for img_file in sorted(img_paths):
        if verbose:
            logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if img.sum() <= 10000:
            if verbose:
                logging.debug(f'Image sum: {img.sum()}')
            logging.warning(f'{img_file} has no data.')
            continue

        # Get the upper and lower points
        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]

        # As the height of a person is larger than the width,
        # use the height to calculate resize ratio.
        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        # Get the median of the x-axis and take it as the person's x-center.
        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if not x_center:
            logging.warning(f'{img_file} has no center.')
            continue

        # Get the left and right points
        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width))
            img = np.concatenate([_, img, _], axis=1)

        to_pickle.append(img[:, left: right].astype('uint8'))

    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        dst_path = os.path.join(output_path, *sinfo)
        # print(img_paths[0].as_posix().split('/'),img_paths[0].as_posix().split('/')[-5])
        # dst_path = os.path.join(output_path, img_paths[0].as_posix().split('/')[-5], *sinfo) if dataset == 'GREW' else dst
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')

    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')

    return [[to_pickle]], [sinfo[0]], [sinfo[1]], [sinfo[2]], np.array([[len(img_groups[1])]])

# 指定数据集的类型CASIAB
def imgs2inputs(input_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB'):
    """Reads a group of images and changes the data in inputs format.

    Args:
        input_path (Path): Image path of silhouette.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    Returns:
        inputs (list): List of Tuple (seqs, labs, typs, vies, seqL) 
    """

    #使用 defaultdict 来存储图像的分组数据，分组依据为 (sid, seq, view)，其中 sid 是主体 ID，seq 是序列 ID，view 是视角信息。该字典的值是图像的路径列表。
    img_groups = defaultdict(list)
    total_files = 0 #一共处理的图像数量
    inputs = [] #存储最终输出的图像数据，格式为元组列表
    sinfos = [] #存储不同 (sid, seq, view) 组合的唯一元组，用于避免重复处理

    #遍历图像路径并分组
    for img_path in input_path.rglob('*.png'):
        #解析图像路径，获取主体，序列，视角信息存储在sinfos，然后使用image_gorup进行分组。
        *_, sid, seq, view, _ = img_path.as_posix().split('/')
        if (sid, seq, view) not in sinfos:
            sinfos.append((sid, seq, view))
        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1
    #对每个 (sid, seq, view) 组合，提取该组合下的所有图像路径，并按文件名进行排序
    for sinfo in sinfos:
        img_paths = img_groups[(sinfo[0], sinfo[1], sinfo[2])]
        to_pickle = []
        for img_file in sorted(img_paths):

            #使用 OpenCV 读取每张图像，并将其转换为灰度图像，因为剪影图像通常是二值化的，灰度图像可以直接用于后续处理。
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

            #如果数据集是 GREW，直接将图像转换为 uint8 类型，并跳过后续处理。
            # 这是因为 GREW 数据集的图像已经符合要求，无需额外裁剪或调整。
            if dataset == 'GREW':
                to_pickle.append(img.astype('uint8'))
                continue
            
            #对于 CASIAB 数据集，首先通过 y_sum 计算图像每行的像素和，
            # 从中找到第一个非零像素行（y_top）和最后一个非零像素行（y_btm），
            # 从而裁剪掉图像中的空白区域，仅保留人物部分。
            # Get the upper and lower points
            y_sum = img.sum(axis=1)
            y_top = (y_sum != 0).argmax(axis=0)
            y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
            img = img[y_top: y_btm + 1, :]


            #调整图像大小
            # As the height of a person is larger than the width,
            # use the height to calculate resize ratio.
            ratio = img.shape[1] / img.shape[0]
            img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

            # Get the median of the x-axis and take it as the person's x-center.
            x_csum = img.sum(axis=0).cumsum()
            x_center = None
            for idx, csum in enumerate(x_csum):
                if csum > img.sum() / 2:
                    x_center = idx
                    break

            if not x_center:
                logging.warning(f'{img_file} has no center.')
                continue

            # Get the left and right points
            half_width = img_size // 2
            left = x_center - half_width
            right = x_center + half_width
            if left <= 0 or right >= img.shape[1]:
                left += half_width
                right += half_width
                _ = np.zeros((img.shape[0], half_width))
                img = np.concatenate([_, img, _], axis=1)

            to_pickle.append(img[:, left: right].astype('uint8'))

        if to_pickle:
            to_pickle = np.asarray(to_pickle)

        if len(to_pickle) < 5:
            logging.warning(f'{sinfo} has less than 5 valid data.')

        """将每个 (sid, seq, view) 组合的处理结果保存为一个元组，并将其添加到 inputs 列表中。每个元组包含：
            处理后的图像序列 ([[to_pickle]])
            主体 ID (sinfo[0])
            序列 ID (sinfo[1])
            视角信息 (sinfo[2])
            图像序列的长度（即该组合下图像的数量）"""
            
        inputs.append(([[to_pickle]], [sinfo[0]], [sinfo[1]], [sinfo[2]],
                      np.array([[len(img_groups[(sinfo[0], sinfo[1], sinfo[2])])]])))
    return inputs


def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False,
             dataset: str = 'CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.

    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    img_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    for img_path in input_path.rglob('*.png'):
        if 'gei.png' in img_path.as_posix():
            continue
        if verbose:
            logging.debug(f'Adding {img_path}')
        *_, sid, seq, view, _ = img_path.as_posix().split('/')
        img_groups[(sid, seq, view)].append(img_path)
        total_files += 1

    logging.info(f'Total files listed: {total_files}')
    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')
    result = []
    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for a in pool.imap_unordered(
                partial(imgs2pickle, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset),
                img_groups.items()):
            progress.update(1)
            result.append(a)
    logging.info('Done')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str,
                        help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w',
                        format='[%(asctime)s - %(levelname)s]: %(message)s')

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size,
             workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
