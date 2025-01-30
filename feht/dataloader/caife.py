import pandas as pd
import numpy as np
from random import randint
from torchvision import transforms
import torch.utils.data as data
from loguru import logger
import os, torch, feht


class CaifeDetectionDataset(data.Dataset):
    def __init__(self, config, phase) -> None:
        assert phase in ['train', 'val', 'test', 'anomaly'], "phase should be one of train/val/test"
        super().__init__()
        self.config = config
        self.phase = phase
        self.input_shape = self.config['input_shape']
        self.__initiate()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            ])
        

    def __initiate(self) -> None:
        logger.info(f">>> Loading {self.phase} meta index...")
        sample_dict = feht.load_pkl(self.config['bbox_index_path'])
        self.sample_list = sample_dict[self.phase]


    def __getitem__(self, index: int) -> dict:
        sample = self.sample_list[index]

        puid = sample['puid']
        suid = sample['suid']
        ori_video_length = len(sample['data_path'])

        clip_length = self.config['clip_length']
        if clip_length > ori_video_length or self.phase != 'train':
            start_point = 0
            video = self.__load_image(sample['data_path'])
        else:
            start_point = randint(0, ori_video_length - clip_length)
            video = self.__load_image(sample['data_path'][start_point: start_point + clip_length])

        if self.phase not in ['train', 'val', 'test']:
            # if is an external prediction，has no gt to load
            data = {
                'puid': puid,
                'suid': suid,
                'ori_video_length': ori_video_length,
                'clip_length': clip_length,
                'shape':Image.open(
                    os.path.join(self.config['data_root'], sample['data_path'][0])
                    ).size[::-1],
            }

            if ori_video_length < clip_length:
                # need padding
                padding = clip_length - ori_video_length
                padding_arr = video[-1].repeat(padding, 1, 1)
                video = torch.cat([video, padding_arr], dim=0)
                video_length = video.shape[0]
            else:
                video_length = ori_video_length

            data['video'] = video.unsqueeze(0)
            data['video_length'] = video_length
            return data

        bbox_anno = sample['annotation_data']
        video_shape = bbox_anno[0]['frame_shape']
        
        center_x, center_y, radius_x, radius_y, presence_class = \
            np.zeros(ori_video_length), np.zeros(ori_video_length), np.zeros(ori_video_length), \
                np.zeros(ori_video_length), np.zeros(ori_video_length), 
        
        for anno in bbox_anno:
            frame_index = anno['frame_index']
            center_x[frame_index] = anno['center'][1] / video_shape[0]
            center_y[frame_index] = anno['center'][0] / video_shape[1]
            radius_x[frame_index] = anno['radius'] / video_shape[1]
            radius_y[frame_index] = anno['radius'] / video_shape[0]
            presence_class[frame_index] = 1

        if 'use_privileged_label' in self.config.keys() and self.config['use_privileged_label']:
            center_x = self.__add_privileged_annotation(center_x)
            center_y = self.__add_privileged_annotation(center_y)
            radius_x = self.__add_privileged_annotation(radius_x)
            radius_y = self.__add_privileged_annotation(radius_y)

        # padding
        if ori_video_length < self.config['clip_length']:
             # padding the video
            padding = self.config['clip_length'] - ori_video_length
            padding_arr = video[-1].repeat(padding, 1, 1)
            video = torch.cat([video, padding_arr], dim=0)
            video_length = video.shape[0]

            # padding the bbox annotation
            center_x = np.pad(center_x, (0, padding), 'constant', constant_values=center_x[-1])
            center_y = np.pad(center_y, (0, padding), 'constant', constant_values=center_y[-1])
            radius_x = np.pad(radius_x, (0, padding), 'constant', constant_values=radius_x[-1])
            radius_y = np.pad(radius_y, (0, padding), 'constant', constant_values=radius_y[-1])
            presence_class = np.pad(presence_class, (0, padding), 'constant', constant_values=presence_class[-1])

        else: 
            video_length = ori_video_length
            
        center = torch.Tensor(np.stack([center_x, center_y], axis=1))
        radius = torch.Tensor(np.stack([radius_x, radius_y], axis=1))
        presence_class = torch.LongTensor(presence_class)

        if self.phase in ['train']:
            data = {
                'video': video.unsqueeze(0),
                'class': presence_class[start_point: start_point + clip_length],
                'center': center[start_point: start_point + clip_length],
                'radius': radius[start_point: start_point + clip_length],
            }
        else:
            data = {
                'video': video.unsqueeze(0),
                'class': presence_class,
                'center': center,
                'radius': radius,
                'shape': video_shape,
                'puid': puid,
                'suid': suid,
                'video_length': video_length,
                'ori_video_length': ori_video_length,
                'clip_length': clip_length
            }
        return data
    

    def __add_privileged_annotation(self, arr: np.ndarray) -> np.ndarray:
        '''
        To leverage the negative cases.
        If the 0 is in the middle of the array, replace as linear interpolation as the nonzero number of its left and right.
        If the 0 is at the beginning, should be replaced by the first nonzero value in the list.
        If the 0 is in the bottom, it should be repalced by the last nonzero value in the list
        '''
        arr = np.array(arr, dtype=float)
        n = len(arr)
        
        non_zero_indices = np.nonzero(arr)[0]
        
        if len(non_zero_indices) == 0:
            return arr
        
        first_non_zero = non_zero_indices[0]
        if first_non_zero > 0:
            arr[:first_non_zero] = arr[first_non_zero]
        
        last_non_zero = non_zero_indices[-1]
        if last_non_zero < n - 1:
            arr[last_non_zero + 1:] = arr[last_non_zero]
        
        zero_indices = np.where(arr == 0)[0]
        
        if len(zero_indices) > 0:
            x = np.arange(n)
            y = arr[non_zero_indices]
            arr[zero_indices] = np.interp(zero_indices, non_zero_indices, y)
        
        return arr


    def __load_image(
            self,
            paths: list[str],
            ) -> torch.Tensor:
        images = []
        for path in paths:
            frame_path = os.path.join(self.config['data_root'], path)
            images.append(self.transform(Image.open(frame_path)))
        return torch.cat(images, dim=0)


    def __len__(self) -> int:
        return len(self.sample_list)