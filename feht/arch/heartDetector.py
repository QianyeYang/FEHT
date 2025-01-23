import torch, os, veronica, cv2
import torch.nn as nn
import numpy as np

from scipy.ndimage import gaussian_filter1d
from icecream import ic
from torch.utils.data import DataLoader
from src.arch.baseArch import BaseArch
from src.dataloader.caife import CaifeDetectionDataset
from tqdm import tqdm
from veronica.model.video import R3D18_WEFLP, R2D18_WEFLP
from veronica.metric.classification import accuracy
from veronica.metric.detection import IoU

from veronica.plot.confusionMatrix import plot_confusion_matrix
from veronica import current_time
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip


class HeartDetector(BaseArch):
    def __init__(self, config):
        super(HeartDetector, self).__init__(config)
        self.parse_config(self.config)

        network_type = self.config.get('network', None)
        if network_type == 'conv2d+channel':
            self.net = R2D18_WEFLP(
                channel_in=self.config['clip_length'], 
                num_out=6
            )
        else:
            self.net = R3D18_WEFLP(num_out=6)
        self.net = self.net.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.w_ce))
        
        self.best_ap50 = 0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.set_lr_scheduler(self.config.get('lr_scheduler', None))
        self.set_dataloader()


    def parse_config(self, config: dict) -> None:
        super().parse_config(config)
        self.num_cpus = config['num_cpus']
        self.w_ce = torch.tensor(config['w_ce']).to(self.device)

    
    def set_dataloader(self) -> None:
        self.train_loader = DataLoader(
            dataset=CaifeDetectionDataset(self.config, phase="train"), 
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            shuffle=True,
            drop_last=True
            )
        self.test_loader = DataLoader(
            dataset=CaifeDetectionDataset(self.config, phase="test"), 
            batch_size=1, 
            num_workers=self.num_cpus,
            shuffle=False,
            drop_last=False
            )
        self.val_loader = DataLoader(
            dataset = CaifeDetectionDataset(self.config, phase="val"),
            batch_size=1,
            num_workers=self.num_cpus,
            shuffle=False
            )


    def forward(self, input_dict:dict) -> torch.Tensor:
        self.optimizer.zero_grad()
        pred_classes, pred_center, pred_radius = self.net(input_dict['video'].to(self.device))
        return {
            'pred_classes': pred_classes,
            'pred_center': pred_center,
            'pred_radius': pred_radius  
            }
    

    def backward(self, input_dict: dict, output_dict: dict) -> None:
        loss = self.loss(input_dict, output_dict)
        if self.phase == 'train':
            loss.backward()
            self.optimizer.step()


    def l2_grad_reg(self, x: torch.Tensor) ->torch.Tensor:
        '''
        L2 gradient regularization
        '''
        grad = x[:, 1:, :] - x[:, :-1, :]
        return torch.mean(grad**2)
    
    
    def adapted_mse_loss(
            self, 
            pred: torch.Tensor,
            gt: torch.Tensor,
            pred_classes: torch.Tensor
            ) -> torch.Tensor:
        mse = (pred - gt) ** 2
        if not self.config['use_privileged_label']:
            pred_classes = torch.argmax(pred_classes.detach(), dim=-1)
            mse *= pred_classes[..., None]
        return torch.mean(mse)
    

    def adapted_mse_loss_gt(
            self, 
            pred: torch.Tensor,
            gt: torch.Tensor,
            gt_classes: torch.Tensor
            ) -> torch.Tensor:
        mse = (pred - gt) ** 2
        if not self.config['use_privileged_label']:
            mse *= gt_classes[..., None]
        weight = self.config.get('w_mse', 10.0)
        return weight * torch.mean(mse)
    
    
    def loss(self, input_dict: dict, output_dict) -> torch.Tensor:
        pred_classes = output_dict['pred_classes'].view(-1, 2)
        gt_classes = input_dict['class'].to(self.device).view(-1)  # [B, clip_length] -> [B*clip_length]

        pred_center = output_dict['pred_center']  # [B, clip_length, 2]
        gt_center = input_dict['center'].to(self.device)
        
        pred_radius = output_dict['pred_radius']  
        gt_radius = input_dict['radius'].to(self.device)

        loss_ce = self.ce_loss(pred_classes, gt_classes)
        pred_classes = torch.argmax(pred_classes.detach(), dim=1)

        # loss_center = 10.0 * self.adapted_mse_loss(pred_center, gt_center, output_dict['pred_classes'])
        # loss_radius = 10.0 * self.adapted_mse_loss(pred_radius, gt_radius, output_dict['pred_classes']) 

        loss_center = self.adapted_mse_loss_gt(pred_center, gt_center, input_dict['class'].to(self.device))
        loss_radius = self.adapted_mse_loss_gt(pred_radius, gt_radius, input_dict['class'].to(self.device))        

        if self.phase == 'train':
            self.logger.info(f"Epoch: {self.epoch}, Iter: {self.iteration+1}/{len(self.train_loader)}")
            self.logger.info(f"Current LR: {self.get_current_lr():.8f}")
            
        elif self.phase == 'val':
            self.logger.info(f"Epoch: {self.epoch}, Iter: {self.iteration+1}/{len(self.val_loader)}")

        self.logger.info(f"Loss_CE: {loss_ce.item():.3f}")
        self.logger.info(f"Loss_center: {loss_center.item():.3f}")
        self.logger.info(f"Loss_radius: {loss_radius.item():.3f}")

        if self.config['w_l2reg'] > 0:
            loss_l2reg = self.config['w_l2reg'] * self.l2_grad_reg(pred_center)
            self.logger.info(f"Loss_L2Reg: {loss_l2reg.item():.3f}")
        else: loss_l2reg = 0

        overall_loss = loss_ce + loss_center + loss_radius + loss_l2reg
        self.logger.info(f"Overall Loss: {overall_loss.item():.3f}")
        self.logger.info(f"Acc.: {accuracy(pred_classes, gt_classes):.3f}")
        self.logger.info("-----------------")
        
        return overall_loss

        
    def train(self) -> None:
        super().train()
        for self.epoch in range(self.num_epochs):
            self.to_train_mode()
            for self.iteration, input_dict in enumerate(self.train_loader):
                output_dict = self.forward(input_dict)
                self.backward(input_dict, output_dict)
            self.scheduler.step()


    @torch.no_grad()
    def validate(self) -> None:
        super().validate()
        ap_calc_predictions, ap_calc_ground_truth = [], []
        for self.iteration, input_dict in enumerate(self.val_loader):

            self.logger.info(
                f">>> Validating on {self.iteration+1}/{len(self.val_loader)} pid: {input_dict['puid']}, scan: {input_dict['suid']}"
                )
            vlength, clength = input_dict['video_length'], input_dict['clip_length']
            stride = int(clength//2) if clength > 1 else 1

            counts = np.zeros(vlength)
            logits = np.zeros((vlength, 2))
            centers = np.zeros((vlength, 2))
            radius = np.zeros((vlength, 2))
            
            for start_idx in tqdm(range(0, vlength-clength+1, stride)):
                end_idx = start_idx + clength
                counts[start_idx:end_idx] += 1
                
                clip = input_dict['video'][:, :, start_idx:end_idx, ...]
                output_dict = self.forward({'video': clip.to(self.device)})  ## fix cuda!!!!!!!!!!!
                
                logits[start_idx:end_idx] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[start_idx:end_idx] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[start_idx:end_idx] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()
            
            if start_idx + clength < vlength:   # deal with the last clip
                counts[vlength-clength:vlength] += 1
                clip = input_dict['video'][:, :, vlength-clength:vlength, ...]
                output_dict = self.forward({'video': clip.to(self.device)})

                logits[vlength-clength:vlength] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[vlength-clength:vlength] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[vlength-clength:vlength] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()

            counts = np.expand_dims(counts, axis=1)
            logits = logits / counts
            centers = centers / counts
            radius = radius / counts

            radius = radius * np.argmax(logits, axis=1)[..., None]
            if np.any(radius != 0):
                radius = np.sum(radius) / (2 * np.sum(np.argmax(logits, axis=1)))
                radius = np.array([[radius, radius]]).repeat(vlength, axis=0)

            formated_prediction, formated_gt = self.format_evaluation(
                frame_shape=input_dict['shape'],
                gt_center=input_dict['center'].numpy().squeeze(),
                gt_radius=input_dict['radius'].numpy().squeeze(),
                gt_classes=input_dict['class'].numpy().squeeze(),
                pred_center=centers,
                pred_radius=radius,
                pred_logits=logits
            )
            ap_calc_predictions.extend(formated_prediction)
            ap_calc_ground_truth.extend(formated_gt)
        
        # SOT performance
        self.logger.info(">>> Calculating the mAP ...")
        ap = MeanAveragePrecision()
        ap.update(ap_calc_predictions, ap_calc_ground_truth)
        results = ap.compute()
        ap50 = results['map_50']
        self.logger.info(f"mAP@50: {ap50:.3f}")
        if ap50 > self.best_ap50:
            self.best_ap50 = ap50
            self.save_state(type='best')


    @torch.no_grad()
    def ext_prediction_on_anomaly_data(self):
        super().inference()
        results_collection = {}

        ext_dataloader = DataLoader(
            dataset=CaifeDetectionDataset(self.config, phase="anomaly"),
            batch_size=1,
            num_workers=self.num_cpus,
            shuffle=False,
            drop_last=False
            )
        
        all_prediction_logits = []
        for self.iteration, input_dict in enumerate(ext_dataloader):
            
            self.logger.info(
                f">>> Predict on {self.iteration+1}/{len(ext_dataloader)} pid: {input_dict['puid']}, scan: {input_dict['suid']}"
                )
            vlength, clength = input_dict['video_length'], input_dict['clip_length']
            stride = int(clength//4) if clength > 1 else 1

            counts = np.zeros(vlength)
            logits = np.zeros((vlength, 2))
            centers = np.zeros((vlength, 2))
            radius = np.zeros((vlength, 2))

            for start_idx in tqdm(range(0, vlength-clength+1, stride)):
                end_idx = start_idx + clength
                counts[start_idx:end_idx] += 1
                
                clip = input_dict['video'][:, :, start_idx:end_idx, ...]
                output_dict = self.forward({'video': clip.to(self.device)})
                
                logits[start_idx:end_idx] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[start_idx:end_idx] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[start_idx:end_idx] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()
            
            if start_idx + clength < vlength:   # deal with the last clip
                counts[vlength-clength:vlength] += 1
                clip = input_dict['video'][:, :, vlength-clength:vlength, ...]
                output_dict = self.forward({'video': clip.to(self.device)})

                logits[vlength-clength:vlength] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[vlength-clength:vlength] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[vlength-clength:vlength] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()

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

            self.build_bbox_predicted_annotation_only_prediction(
                videos=input_dict['video'].numpy().squeeze()[:ori_vlength],
                pred_center=centers,
                pred_radius=radius,
                pred_classes=logits,
                frame_shape=(int(input_dict['shape'][0]), int(input_dict['shape'][1])),
                save_path=os.path.join(self.vis_dir, 'ext_anomalies_pred', f'{input_dict["suid"]}.mp4')
            )

            # save qualitative results
            final_results = {
                'logits': logits,
                'radius': radius,
                'center': centers,
                'puid': input_dict['puid'],
                'suid': input_dict['suid']
            }
            results_collection[input_dict['suid'][0]] = final_results
            sample_save_path = os.path.join(self.res_dir, 'ext_anomalies_pred', f'{input_dict["suid"][0]}.pkl')
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            veronica.save_pkl(final_results, sample_save_path, overwrite=True)

        # check if it is able to save as a pickle
        veronica.save_pkl(
            results_collection, 
            os.path.join(self.res_dir, 'ext_anomalies_pred', f'ext_prediction_on_anomaly_data.pkl'),
            overwrite=True
            )



    @torch.no_grad()
    def inference(self) -> None:
        super().inference()

        results_collection = {}

        all_prediction_logits, all_ground_truth = [], []
        ap_calc_predictions, ap_calc_ground_truth = [], []

        for self.iteration, input_dict in enumerate(self.test_loader):

            self.logger.info(
                f">>> Testing on {self.iteration+1}/{len(self.test_loader)} pid: {input_dict['puid']}, scan: {input_dict['suid']}"
                )
            vlength, clength = input_dict['video_length'], input_dict['clip_length']
            ori_vlength = input_dict['ori_video_length']
            stride = stride = int(clength//4) if clength > 1 else 1

            counts = np.zeros(vlength)
            logits = np.zeros((vlength, 2))
            centers = np.zeros((vlength, 2))
            radius = np.zeros((vlength, 2))
            
            for start_idx in tqdm(range(0, vlength-clength+1, stride)):
                end_idx = start_idx + clength
                counts[start_idx:end_idx] += 1
                
                clip = input_dict['video'][:, :, start_idx:end_idx, ...]
                output_dict = self.forward({'video': clip.to(self.device)})
                
                logits[start_idx:end_idx] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[start_idx:end_idx] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[start_idx:end_idx] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()
            
            if start_idx + clength < vlength:   # deal with the last clip
                counts[vlength-clength:vlength] += 1
                clip = input_dict['video'][:, :, vlength-clength:vlength, ...]
                output_dict = self.forward({'video': clip.to(self.device)})

                logits[vlength-clength:vlength] += output_dict['pred_classes'].detach().cpu().squeeze().numpy()
                centers[vlength-clength:vlength] += output_dict['pred_center'].detach().cpu().squeeze().numpy()
                radius[vlength-clength:vlength] += output_dict['pred_radius'].detach().cpu().squeeze().numpy()

            # print(f"counts: {counts}")
            # print(f"before devide: {radius}")
            counts = np.expand_dims(counts, axis=1)
            logits = logits / counts
            centers = centers / counts
            radius = radius / counts
            # print(f"after devide: {radius}")


            all_prediction_logits.append(logits)
            all_ground_truth.append(input_dict['class'].numpy().squeeze())

            # self.build_bbox_predicted_annotation(
            #     videos=input_dict['video'].numpy().squeeze(),
            #     gt_center=input_dict['center'].numpy().squeeze(),
            #     gt_radius=input_dict['radius'].numpy().squeeze(),
            #     gt_classes=input_dict['class'].numpy().squeeze(),
            #     pred_center=centers,
            #     pred_radius=radius,
            #     pred_classes=logits,
            #     frame_shape=(int(input_dict['shape'][0]), int(input_dict['shape'][1])),
            #     save_path=os.path.join(self.vis_dir, f'{input_dict["suid"]}.mp4')
            # )

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

            suid = input_dict['suid'][0]
            results_collection[suid] = self.calc_sample_results(
                frame_shape=input_dict['shape'],
                pred_radius=radius,
                pred_center=centers,
                pred_logits=logits,
                gt_radius=input_dict['radius'].numpy().squeeze(),
                gt_center=input_dict['center'].numpy().squeeze(),
                gt_classes=input_dict['class'].numpy().squeeze()
            )
            
            formated_prediction, formated_gt = self.format_evaluation(
                frame_shape=input_dict['shape'],
                gt_center=input_dict['center'].numpy().squeeze(),
                gt_radius=input_dict['radius'].numpy().squeeze(),
                gt_classes=input_dict['class'].numpy().squeeze(),
                pred_center=centers,
                pred_radius=radius,
                pred_logits=logits
            )
            ap_calc_predictions.extend(formated_prediction)
            ap_calc_ground_truth.extend(formated_gt)
            
            # self.build_bbox_predicted_annotation(
            #     videos=input_dict['video'].numpy().squeeze(),
            #     gt_center=input_dict['center'].numpy().squeeze(),
            #     gt_radius=input_dict['radius'].numpy().squeeze(),
            #     gt_classes=input_dict['class'].numpy().squeeze(),
            #     pred_center=centers,
            #     pred_radius=radius,
            #     pred_classes=logits,
            #     frame_shape=(int(input_dict['shape'][0]), int(input_dict['shape'][1])),
            #     save_path=os.path.join(self.vis_dir, f'{input_dict["suid"]}_sigma_{sigma}.mp4')
            # )

        all_prediction_logits = np.concatenate(all_prediction_logits)
        all_ground_truth = np.concatenate(all_ground_truth)
        all_prediction = torch.argmax(torch.tensor(all_prediction_logits), dim=1)
        all_prediction = all_prediction.numpy().tolist()

        acc = accuracy(
            all_prediction,
            all_ground_truth
            )
        self.logger.info(f">>> Test accuracy: {acc:.3f}")

        self.logger.info(">>> Calculating the mse and IoUs ...")
        mse_center = np.mean([i['mse_center'] for _, i in results_collection.items()])
        std_center = np.std([i['mse_center'] for  _,i in results_collection.items()])
        mse_radius = np.mean([i['mse_radius'] for _,i in results_collection.items()])
        std_radius = np.std([i['mse_radius'] for _,i in results_collection.items()])
        ious = np.mean([i['IoU'] for _,i in results_collection.items()])
        std_ious = np.std([i['IoU'] for _,i in results_collection.items()])

        self.logger.info(f"mse_center: {mse_center:.3f} +/- {std_center:.3f}")
        self.logger.info(f"mse_radius: {mse_radius:.3f} +/- {std_radius:.3f}")
        self.logger.info(f"IoU: {ious:.3f} +/- {std_ious:.3f}")
        
        veronica.save_pkl(
            results_collection,
            os.path.join(self.res_dir, f'results.pkl'),
            overwrite=True
            )

        # SOT performance
        self.logger.info(">>> Calculating the mAP ...")
        ap = MeanAveragePrecision()
        ap.update(ap_calc_predictions, ap_calc_ground_truth)
        results = ap.compute()
        self.logger.info(results)

        ap_55 = MeanAveragePrecision(iou_thresholds=[0.55])
        ap_55.update(ap_calc_predictions, ap_calc_ground_truth)
        self.logger.info(">>> mAP@55")
        self.logger.info(ap_55.compute())

        ap_60 = MeanAveragePrecision(iou_thresholds=[0.60])
        ap_60.update(ap_calc_predictions, ap_calc_ground_truth)
        self.logger.info(">>> mAP@60")
        self.logger.info(ap_60.compute())

        ap_65 = MeanAveragePrecision(iou_thresholds=[0.65])
        ap_65.update(ap_calc_predictions, ap_calc_ground_truth)
        self.logger.info(">>> mAP@65")
        self.logger.info(ap_65.compute())

        ap_70 = MeanAveragePrecision(iou_thresholds=[0.70])
        ap_70.update(ap_calc_predictions, ap_calc_ground_truth)
        self.logger.info(">>> mAP@70")
        self.logger.info(ap_70.compute())

        
    #     self.logger.info(">>> Saving the predictions")
    #     results = []
    #     for i in range(len(image_paths)):
    #         sample_results = {
    #             'image_path': image_paths[i],
    #             'prediction': str(all_prediction[i]), ###
    #             'ground_truth': str(all_ground_truth[i])
    #         }
    #         results.append(sample_results)
    #     veronica.save_json(
    #         data = results, 
    #         path = os.path.join(self.res_dir, f'predictions_{self.exp_name}.json'),
    #         overwrite=True
    #         )
        
        self.logger.info(">>> Saving the visualisation")
        # ic(len(np.unique(all_ground_truth)))
        plot_confusion_matrix(
            y_true=all_ground_truth,
            y_pred=all_prediction,
            target_names=['wo/heart', 'w/heart'],
            title=f"Confusion Matrix of Expt:{self.exp_name}",
            save_path = os.path.join(self.res_dir, f'confusion_matrix_{self.exp_name}.png')
        )


    def build_bbox_predicted_annotation(
        self, 
        videos: list,
        gt_center: list,
        gt_radius: list,
        gt_classes: list,
        pred_center: list,
        pred_radius: list,
        pred_classes : list,
        frame_shape: list|tuple,
        save_path: str
        ) -> list:
        '''
        Build the predicted annotation for the bounding box.
        '''
        vis_results = []

        for idx, v_arr in enumerate(videos):
            # get the shape of the video
            pcx, pcy = pred_center[idx]
            pcy, pcx = int(pcx * frame_shape[0]), int(pcy * frame_shape[1])
            prx, pry = pred_radius[idx]
            prx, pry = int(prx * frame_shape[1]), int(pry * frame_shape[0])
            prx = pry = int((prx + pry) / 2)
            
            gcx, gcy = gt_center[idx]
            gcy, gcx = int(gcx * frame_shape[0]), int(gcy * frame_shape[1])
            grx, gry = gt_radius[idx]
            grx, gry = int(grx * frame_shape[1]), int(gry * frame_shape[0])

            # resize to original size
            v_arr = (v_arr*255).astype(np.uint8)
            v_arr = cv2.resize(v_arr, (frame_shape[1], frame_shape[0]))
            v_arr = cv2.cvtColor(v_arr, cv2.COLOR_GRAY2RGB)

            
            gt_cls = gt_classes[idx]
            pd_cls = np.argmax(pred_classes[idx])

            DRAW_BBOX = True

            if gt_cls == pd_cls == 1:
                # text put on lefttop
                cv2.putText(
                    v_arr, 
                    f'Ture Positive', 
                    (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                    )
            elif gt_cls == pd_cls == 0:
                DRAW_BBOX = False
                cv2.putText(
                    v_arr, 
                    f'Ture Negative', 
                    (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                    )
            elif gt_cls == 1 and pd_cls == 0:
                DRAW_BBOX = False
                cv2.putText(
                    v_arr, 
                    f'False Negative', 
                    (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2
                    )
            elif gt_cls == 0 and pd_cls == 1:
                cv2.putText(
                    v_arr, 
                    f'False Positive', 
                    (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2
                    )
            else:
                cv2.putText(
                    v_arr, 
                    f'Unknown', 
                    (10, 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2)

            # draw gt box with green 
            if gcx !=0 and gcy != 0:
                cv2.rectangle(
                    v_arr, 
                    (gcx-grx, gcy-gry), 
                    (gcx+grx, gcy+gry), 
                    (0, 255, 0), 2
                    )
                
            if pcx !=0 and pcy != 0 and DRAW_BBOX:
                # draw pred box with red
                cv2.rectangle(
                    v_arr, 
                    (pcx-prx, pcy-pry), 
                    (pcx+prx, pcy+pry), 
                    (0, 0, 255), 2
                    )
            
            v_arr = cv2.cvtColor(v_arr, cv2.COLOR_BGR2RGB)
            vis_results.append(v_arr)

        vis_results = [np.array(i, dtype=np.uint8) for i in vis_results]
        vclip = ImageSequenceClip(vis_results, fps=50)
        vclip.write_videofile(
            save_path,
            codec="libx264",
            fps=50
            )
        
    def build_bbox_predicted_annotation_only_prediction(
        self, 
        videos: list,
        pred_center: list,
        pred_radius: list,
        pred_classes : list,
        frame_shape: list|tuple,
        save_path: str
        ) -> list:
        '''
        Build the predicted annotation for the bounding box.
        '''
        vis_results = []
        

        for idx, v_arr in enumerate(videos):
            # get the shape of the video
            pcx, pcy = pred_center[idx]
            pcy, pcx = int(pcx * frame_shape[0]), int(pcy * frame_shape[1])
            prx, pry = pred_radius[idx]
            prx, pry = int(prx * frame_shape[1]), int(pry * frame_shape[0])
            prx = pry = int((prx + pry) / 2)
            
            # resize to original size
            v_arr = (v_arr*255).astype(np.uint8)
            v_arr = cv2.resize(v_arr, (frame_shape[1], frame_shape[0]))
            v_arr = cv2.cvtColor(v_arr, cv2.COLOR_GRAY2RGB)

            pd_cls = np.argmax(pred_classes[idx])

            if pd_cls == 1:
                cv2.rectangle(
                    v_arr, 
                    (pcx-prx, pcy-pry), 
                    (pcx+prx, pcy+pry), 
                    (0, 255, 255), 2
                    )

            v_arr = cv2.cvtColor(v_arr, cv2.COLOR_BGR2RGB)
            vis_results.append(v_arr)

        vis_results = [np.array(i, dtype=np.uint8) for i in vis_results]
        vclip = ImageSequenceClip(vis_results, fps=50)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vclip.write_videofile(
            save_path,
            codec="libx264",
            fps=50
            )


    def calc_sample_results(
            self,
            frame_shape: list|tuple,
            gt_center: np.ndarray, 
            gt_radius: np.ndarray,
            gt_classes: np.ndarray,
            pred_center: np.ndarray,
            pred_radius: np.ndarray,
            pred_logits: np.ndarray
        ) -> dict:
        
        pred_center_restored = []
        pred_radius_restored = []
        gt_center_restored = []
        gt_radius_restored = []
        IoU_res = []
        pred_boxes = []
        gt_boxes = []
        
        for idx in range(pred_center.shape[0]):
            pcx, pcy = pred_center[idx]
            pcy, pcx = int(pcx * frame_shape[0]), int(pcy * frame_shape[1])
            pred_center_restored.append([pcx, pcy])

            prx, pry = pred_radius[idx]
            # print(f"prx: {prx}, pry: {pry}")
            # print(f"frame_shape: {frame_shape}")
            prx, pry = int(prx * frame_shape[1]), int(pry * frame_shape[0])
            prx = pry = int((prx + pry) / 2)
            pred_radius_restored.append([prx, pry])
            
            gcx, gcy = gt_center[idx]
            gcy, gcx = int(gcx * frame_shape[0]), int(gcy * frame_shape[1])
            gt_center_restored.append([gcx, gcy])

            grx, gry = gt_radius[idx]
            grx, gry = int(grx * frame_shape[1]), int(gry * frame_shape[0])
            gt_radius_restored.append([grx, gry])

            pred_boxes.append([pcx-prx, pcy-pry, pcx+prx, pcy+pry])
            gt_boxes.append([gcx-grx, gcy-gry, gcx+grx, gcy+gry])


        pred_center = np.array(pred_center_restored)
        pred_radius = np.array(pred_radius_restored)
        gt_center = np.array(gt_center_restored)
        gt_radius = np.array(gt_radius_restored)

        se_center_array = (pred_center - gt_center) ** 2
        se_radius_array = (pred_radius - gt_radius) ** 2

        pred_classes = np.argmax(pred_logits, axis=1)

        for idx in range(len(pred_classes)):
            if pred_classes[idx] == gt_classes[idx] == 1:
                IoU_res.append(IoU(pred_boxes[idx], gt_boxes[idx]))
            elif pred_classes[idx] == gt_classes[idx] == 0:
                se_center_array[idx] = np.array([0, 0])
                se_radius_array[idx] = np.array([0, 0])
                IoU_res.append(1.0)
            elif pred_classes[idx] == 1 and gt_classes[idx] == 0:
                IoU_res.append(0.0)
            elif pred_classes[idx] == 0 and gt_classes[idx] == 1:
                se_center_array[idx] = (gt_center[idx])**2
                IoU_res.append(0.0)

        mse_center = np.mean(se_center_array)
        mse_radius = np.mean(se_radius_array)
        IoU_res = np.mean(IoU_res)

        return {
            'mse_center': mse_center,
            'mse_radius': mse_radius,
            'IoU': IoU_res,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes,
            'pred_classes': pred_classes,
            'gt_classes': gt_classes,
            'frame_shape': frame_shape
        }
        

    def format_evaluation(
            self,
            frame_shape: list|tuple,
            gt_center: np.ndarray, 
            gt_radius: np.ndarray,
            gt_classes: np.ndarray,
            pred_center: np.ndarray,
            pred_radius: np.ndarray,
            pred_logits: np.ndarray
            ) -> list:
        '''
        convert the format to match the "torchmetrics.detection.mean_ap import MeanAveragePrecision"

        1.need to restore the relative y value on pred radius, for using torchmetric for calculation.
        w
        '''
        predictions, targets = [], []

        pred_center_restored = []
        pred_radius_restored = []
        gt_center_restored = []
        gt_radius_restored = []

        for idx in range(pred_center.shape[0]):
            pcx, pcy = pred_center[idx]
            pcy, pcx = int(pcx * frame_shape[0]), int(pcy * frame_shape[1])
            pred_center_restored.append([pcx, pcy])

            prx, pry = pred_radius[idx]
            prx, pry = int(prx * frame_shape[1]), int(pry * frame_shape[0])
            prx = pry = int((prx + pry) / 2)
            pred_radius_restored.append([prx, pry])
            
            gcx, gcy = gt_center[idx]
            gcy, gcx = int(gcx * frame_shape[0]), int(gcy * frame_shape[1])
            gt_center_restored.append([gcx, gcy])

            grx, gry = gt_radius[idx]
            grx, gry = int(grx * frame_shape[1]), int(gry * frame_shape[0])
            gt_radius_restored.append([grx, gry])

        pred_center = np.array(pred_center_restored)
        pred_radius = np.array(pred_radius_restored)
        gt_center = np.array(gt_center_restored)
        gt_radius = np.array(gt_radius_restored)

        pred_box_x1y1 = pred_center - pred_radius
        pred_box_x2y2 = pred_center + pred_radius
        pred_box = np.concatenate([pred_box_x1y1, pred_box_x2y2], axis=1)  # [B, 4]

        gt_box_x1y1 = gt_center - gt_radius
        gt_box_x2y2 = gt_center + gt_radius
        gt_box = np.concatenate([gt_box_x1y1, gt_box_x2y2], axis=1)

        for i in range(len(gt_box)):
            if np.argmax(pred_logits[i]) != 0:
                predictions.append({
                    'boxes': torch.tensor([pred_box[i]]),
                    'scores': torch.tensor([pred_logits[i, 1]]),
                    'labels': torch.tensor([np.argmax(pred_logits[i])], dtype=torch.int64)
                })
            else:
                predictions.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32),
                    'scores': torch.empty((0),),
                    'labels': torch.empty((0), dtype=torch.int64)
                })
            
            if gt_classes[i] != 0:
                targets.append({
                    'boxes': torch.tensor([gt_box[i]]),
                    'labels': torch.tensor([gt_classes[i]], dtype=torch.int64)
                })
            else:
                targets.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32),
                    'labels': torch.empty((0), dtype=torch.int64)
                })

        return predictions, targets
        
        
            

            

        
                    
        