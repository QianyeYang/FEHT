import numpy as np

def IoU(
    gt_bbox: list | np.ndarray,
    pred_bbox: list | np.ndarray
    ) -> float: 
    """
    Calculate the IoU of two bounding boxes.
    boxA and gt_bbox are lists or tuples of [x1, y1, x2, y2].

    :param gt_bbox: ground truth bounding box
    :type gt_bbox: list | np.ndarray
    :param pred_bbox: predicted bounding box
    :type pred_bbox: list | np.ndarray
    """

    if gt_bbox is None and pred_bbox is None:
        return 1.0
    
    if gt_bbox is not None and pred_bbox is None:
        return 0.0
    
    if gt_bbox is None and pred_bbox is not None:
        return 0.0

    xA = max(pred_bbox[0], gt_bbox[0])
    yA = max(pred_bbox[1], gt_bbox[1])
    xB = min(pred_bbox[2], gt_bbox[2])
    yB = min(pred_bbox[3], gt_bbox[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    pred_bbox_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_bbox_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    iou = inter_area / float(pred_bbox_area + gt_bbox_area - inter_area)
    return iou