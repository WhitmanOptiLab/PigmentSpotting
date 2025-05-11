import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import numpy as np


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    """
    how do I know the number of iterations?
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types
         

class DistanceKeypointEvaluator:
    def __init__(self, threshold=5.0, num_keypoints=4):
        self.threshold = threshold
        self.num_keypoints = num_keypoints
        self.correct_counts = np.zeros(num_keypoints, dtype=np.int32)
        self.visible_counts = np.zeros(num_keypoints, dtype=np.int32)
        self.distances = [[] for _ in range(num_keypoints)]  # for mean error tracking
        self.errors = []


    def update(self, predictions, targets):
        for pred, tgt in zip(predictions, targets):
            pred_kps = pred["keypoints"].cpu().numpy()  # shape: (N, K, 3)
            tgt_kps = tgt["keypoints"].cpu().numpy()

            for pred_obj, tgt_obj in zip(pred_kps, tgt_kps):  # Loop over objects
                for k in range(self.num_keypoints):
                    if tgt_obj[k, 2] > 0:  # Keypoint is visible
                        pred_xy = pred_obj[k, :2]
                        tgt_xy = tgt_obj[k, :2]
                        dist = np.linalg.norm(pred_xy - tgt_xy)
                        self.distances[k].append(dist)
                        self.visible_counts[k] += 1
                        if dist < self.threshold:
                            self.correct_counts[k] += 1

    def compute(self):
        accuracy_per_kp = self.correct_counts / np.maximum(self.visible_counts, 1)
        mean_error_per_kp = [np.mean(d) if d else 0.0 for d in self.distances]
        return accuracy_per_kp, mean_error_per_kp

    def summarize(self):
        accs, errors = self.compute()
        self.errors = errors
        print("\nPer-Keypoint Evaluation Summary:")
        for i, (acc, err) in enumerate(zip(accs, errors)):
            print(f"  Keypoint {i}: Accuracy = {acc:.4f}, Mean Error = {err:.2f} px")
        return accs, errors



@torch.inference_mode()
def evaluate(model, data_loader, device):

    model.eval()
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    evaluator = DistanceKeypointEvaluator(threshold=10.0)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets_cpu = [{k: v for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        outputs = model(images)
        outputs_cpu = [{k: v for k, v in o.items()} for o in outputs]

        model_time = time.time() - model_time
        evaluator.update(outputs_cpu, targets_cpu)

        metric_logger.update(model_time=model_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    evaluator.summarize()
    return evaluator


# @torch.inference_mode()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator
