import argparse, os, feht, torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from scipy.ndimage import gaussian_filter1d
from feht.utils.time import current_time
from feht.dataloader.caife import CSVInferenceDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser("Higest level configuration settings")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="assign gpu device", required=False)
    parser.add_argument("-m", "--model", type=str, default='feht-l2-pretrained', help="name of the model need to use", required=False)
    parser.add_argument("-c", "--csv", default='', help="csv index for ", required=True)
    parser.add_argument("-o", "--output", default='', help="output path", required=False)
    args = parser.parse_args()

    # init variables
    results_collection = {}
    all_prediction_logits = []

    # sanity checks
    assert os.path.isfile(args.csv), \
        f"CSV file {args.csv} not found! Please check the file."
    assert args.csv.endswith('.csv'), \
        f"CSV file {args.csv} is not in csv format! Please check the file."
    
    if args.output == '':
        args.output = "./temp_output" + current_time()
        logger.info(f">>> Output path is not specified. Saving the output to {args.output}")
    os.makedirs(args.output, exist_ok=True)
    if len(os.listdir(args.output)) > 0:
        logger.warning(f">>> Output directory {args.output} is not empty! Files may be overwritten.")

    # load the model
    device = torch.device(f"cuda:{args.gpu}")
    clip_length = feht.get_clip_length_from_model(args.model)
    model = feht.Model(name=args.model)
    model = model.to(device)

    # set dataloader
    test_loader = DataLoader(
        dataset=CSVInferenceDataset(csv_index=args.csv),
        batch_size=1,
        num_workers=0,
        shuffle=False,
        )
    
    for iteration, input_dict in enumerate(test_loader):
            
        logger.info(
            f">>> Predict on {iteration+1}/{len(test_loader)} pid: {input_dict['puid']}, scan: {input_dict['suid']}"
            )

        vlength, clength = input_dict['video_length'], clip_length

        # repeat the input_dict['video'] last frame for clip_length times, [batch, channel, time, height, width]
        # to make sure the video length is larger than the clip length
        last_frame = input_dict['video'][:, :, -1, ...].repeat(1, 1, clength, 1, 1)
        input_dict['video'] = torch.cat([input_dict['video'], last_frame], dim=2)
        vlength += clength
        stride = int(clength//4) if clength > 1 else 1

        counts = np.zeros(vlength)
        logits = np.zeros((vlength, 2))
        centers = np.zeros((vlength, 2))
        radius = np.zeros((vlength, 2))

        for start_idx in tqdm(range(0, vlength-clength+1, stride)):
            end_idx = start_idx + clength
            counts[start_idx:end_idx] += 1
            
            clip = input_dict['video'][:, :, start_idx:end_idx, ...]
            pred_classes, pred_center, pred_radius = model(clip.to(device))
            
            logits[start_idx:end_idx] += pred_classes.detach().cpu().squeeze().numpy()
            centers[start_idx:end_idx] += pred_center.detach().cpu().squeeze().numpy()
            radius[start_idx:end_idx] += pred_radius.detach().cpu().squeeze().numpy()
        
        if start_idx + clength < vlength:   # deal with the last clip
            counts[vlength-clength:vlength] += 1
            clip = input_dict['video'][:, :, vlength-clength:vlength, ...]
            pred_classes, pred_center, pred_radius = model(clip.to(device))

            logits[vlength-clength:vlength] += pred_classes.detach().cpu().squeeze().numpy()
            centers[vlength-clength:vlength] += pred_center.detach().cpu().squeeze().numpy()
            radius[vlength-clength:vlength] += pred_radius.detach().cpu().squeeze().numpy()

        counts = np.expand_dims(counts, axis=1)
        logits = logits / counts
        centers = centers / counts
        radius = radius / counts

        all_prediction_logits.append(logits)

        '''
        Do post processing for the prediction.
        1. gaussian filter the center point
        2. use the average value of the radius (when the heart is predicted)
        '''
        sigma = 3
        centers = np.column_stack([
            gaussian_filter1d(centers[:, 0], sigma=sigma),
            gaussian_filter1d(centers[:, 1], sigma=sigma)
        ])

        radius = radius * np.argmax(logits, axis=1)[..., None]
        if np.any(radius != 0): 
            radius = np.sum(radius) / (2 * np.sum(np.argmax(logits, axis=1)))
            radius = np.array([[radius, radius]]).repeat(vlength, axis=0)

        ori_vlength = input_dict['ori_video_length']
        centers = centers[:ori_vlength]
        radius = radius[:ori_vlength]
        logits = logits[:ori_vlength]

        # save qualitative results
        final_results = {
            'logits': logits,
            'radius': radius,
            'center': centers,
            'puid': input_dict['puid'],
            'suid': input_dict['suid'],
        }
        results_collection[input_dict['suid'][0]] = final_results
        sample_save_path = os.path.join(args.output, f'{input_dict["suid"][0]}.pkl')
        feht.save_pkl(final_results, sample_save_path, overwrite=True)
    
    
    

