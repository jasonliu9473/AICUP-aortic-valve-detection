from ensemble_boxes import *
from collections import defaultdict

def read_merged_txt_file(file_path, img_width=512, img_height=512):
    """
    Read a single merged txt file containing predictions for all patients.
    
    Args:
        file_path: Path to merged txt file
        img_width: Image width for normalization
        img_height: Image height for normalization
    
    Returns:
        Dictionary mapping patient_id to their predictions (normalized coordinates)
    """
    predictions = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                patient_id = parts[0]
                label = int(parts[1])
                score = float(parts[2])
                x1 = float(parts[3])
                y1 = float(parts[4])
                x2 = float(parts[5])
                y2 = float(parts[6])
                
                # Normalize coordinates to [0, 1] for WBF
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                
                # Clip to valid range and skip invalid boxes
                x1_norm = max(0.0, min(1.0, x1_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))
                x2_norm = max(0.0, min(1.0, x2_norm))
                y2_norm = max(0.0, min(1.0, y2_norm))
                
                # Skip boxes with zero area
                if x2_norm > x1_norm and y2_norm > y1_norm:
                    predictions[patient_id]['boxes'].append([x1_norm, y1_norm, x2_norm, y2_norm])
                    predictions[patient_id]['scores'].append(score)
                    predictions[patient_id]['labels'].append(label)
    
    return predictions

def ensemble_all_predictions(pred_list, weights=None, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Ensemble predictions from multiple dictionaries using WBF all at once.
    
    Args:
        pred_list: List of dictionaries with patient predictions (normalized coords)
        weights: List of weights for each prediction dict (if None, uses equal weights)
        iou_thr: IoU threshold for fusion
        skip_box_thr: Minimum confidence threshold
    
    Returns:
        Dictionary of ensembled predictions (normalized coords)
    """
    ensembled = {}
    
    # Get all unique patient IDs across all predictions
    all_patients = set()
    for pred in pred_list:
        all_patients.update(pred.keys())
    
    # Set default weights if not provided
    if weights is None:
        weights = [1] * len(pred_list)
    
    for patient_id in all_patients:
        boxes_list = []
        scores_list = []
        labels_list = []
        patient_weights = []
        
        # Collect predictions from all sources for this patient
        for pred, weight in zip(pred_list, weights):
            if patient_id in pred and pred[patient_id]['boxes']:
                boxes_list.append(pred[patient_id]['boxes'])
                scores_list.append(pred[patient_id]['scores'])
                labels_list.append(pred[patient_id]['labels'])
                patient_weights.append(weight)
        
        if not boxes_list:
            continue
        
        # Apply WBF with all models at once
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=patient_weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type='avg'
        )
        
        ensembled[patient_id] = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'labels': labels.astype(int).tolist()
        }
    
    return ensembled

def save_predictions(predictions, output_file, img_width=640, img_height=640):
    """
    Save predictions to txt file in pixel coordinates.
    Args:
        predictions: list of the averaged result
        output_file: output '.txt; file name
        img_width: width of the image
        img_height: height of the image
    """
    with open(output_file, 'w') as f:
        for patient_id in sorted(predictions.keys()):
            preds = predictions[patient_id]
            boxes = preds['boxes']
            scores = preds['scores']
            labels = preds['labels']
            
            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                label = labels[i]
                
                # Denormalize coordinates back to pixels
                x1 = int(round(box[0] * img_width))
                y1 = int(round(box[1] * img_height))
                x2 = int(round(box[2] * img_width))
                y2 = int(round(box[3] * img_height))
                
                f.write(f"{patient_id} {int(label)} {score:.4f} {x1} {y1} {x2} {y2}\n")