import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models.segmentation as models

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 19

def make_fcn_model(weights_path: str) -> nn.Module:
    m = models.fcn_resnet50(weights=None, aux_loss=True)
    m.classifier[4] = nn.Conv2d(512, n_classes, kernel_size=1)
    state = torch.load(weights_path, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

cnn_variants = {
    "base":         make_fcn_model("/home/mlee2/163_project/model_info/best_fcn_resnet50.pth"),
    "base_crf":     make_fcn_model("/home/mlee2/163_project/model_info/best_fcn_resnet50_crf.pth"),
    "grounded":     make_fcn_model("/home/mlee2/163_project/model_info/best_fcn_resnet50_gnd.pth"),
    "grounded_crf": make_fcn_model("/home/mlee2/163_project/model_info/best_fcn_resnet50_gndcrf.pth"),
}

preprocess = transforms.Compose([
    transforms.Resize((512, 1024), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

sam_weights = "/home/mlee2/163_project/model_info/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_weights).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_checkpoint = "/home/mlee2/163_project/model_info/sam2.1_hiera_large.pt"
sam2_model = build_sam2(config_file, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generatorv2 = SAM2AutomaticMaskGenerator(sam2_model,
                                              points_per_side=32,
                                              pred_iou_thresh=0.7,
                                              stability_score_thresh=0.9,
                                              min_mask_region_area=50)
CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
]

def predict_cnn_seg(model, img: Image.Image) -> np.ndarray:
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)["out"]
        pred = out.argmax(dim=1)[0].cpu().numpy()
    return pred

def dense_crf_postprocess(image_np, probs_np, n_iter=5):
    H, W = image_np.shape[:2]
    C = probs_np.shape[0]
    d = dcrf.DenseCRF2D(W, H, C)
    unary = unary_from_softmax(probs_np)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_np, compat=10)
    Q = d.inference(n_iter)
    labels = np.array(Q).reshape((C, H, W)).argmax(axis=0)
    return labels

def predict_cnn_crf_seg(model, img: Image.Image) -> np.ndarray:
    img_resized = img.resize((1024, 512), Image.BILINEAR)
    img_t = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)
    img_np = (np.array(img_resized)).astype(np.uint8)

    img_t_norm = preprocess(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t_norm)["out"]
        probs = F.softmax(out, dim=1)[0].cpu().numpy()  # CxHxW

    labels = dense_crf_postprocess(img_np, probs)
    return labels

def denormalize_image_tensor(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def map_labeled_masks(labeled_masks, H, W):
    pred_map = np.full((H, W), 255, dtype=np.uint8)
    for mask_dict, cls_idx in labeled_masks:
        seg = mask_dict["segmentation"]
        pred_map[seg] = cls_idx
    return pred_map

def predict_cnn_sam_seg(model, img: Image.Image, mg) -> np.ndarray:
    img_resized = img.resize((1024, 512), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.uint8)

    img_t = preprocess(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)["out"]
        probs = F.softmax(out, dim=1)[0]

    masks = mg.generate(img_np)

    labeled_masks = []
    for m in masks:
        seg = m["segmentation"]
        region_probs = probs[:, seg]
        if region_probs.numel() == 0:
            continue
        mean_probs = region_probs.mean(dim=1)
        cls_idx = int(mean_probs.argmax().item())
        labeled_masks.append((m, cls_idx))

    H, W = img_np.shape[:2]
    pred_map = map_labeled_masks(labeled_masks, H, W)
    return pred_map

def predict_cnn_sam_v1_seg(model, img: Image.Image) -> np.ndarray:
    return predict_cnn_sam_seg(model, img, mask_generator)

def predict_cnn_sam_v2_seg(model, img: Image.Image) -> np.ndarray:
    return predict_cnn_sam_seg(model, img, mask_generatorv2)

# decision rule

def decide_action_from_seg(seg_mask: np.ndarray) -> dict:
    h, w = seg_mask.shape

    y1, y2 = int(0.6 * h), h
    x1, x2 = int(0.4 * w), int(0.6 * w)
    roi = seg_mask[y1:y2, x1:x2]

    vals, counts = np.unique(roi, return_counts=True)
    class_counts = {int(v): int(c) for v, c in zip(vals, counts) if v != 255}

    readable_counts = {
        CITYSCAPES_CLASS_NAMES[v]: c
        for v, c in class_counts.items()
        if 0 <= v < len(CITYSCAPES_CLASS_NAMES)
    }

    obstacle_names = ["person", "pole"]
    obstacle_ids = [
        CITYSCAPES_CLASS_NAMES.index(name)
        for name in obstacle_names
        if name in CITYSCAPES_CLASS_NAMES
    ]

    obstacle_present = any(c in class_counts for c in obstacle_ids)

    if obstacle_present:
        action = "turn_left"
    else:
        action = "forward"

    return {
        "action": action,
        "roi_counts": readable_counts,
    }

# FastAPI connection

app = FastAPI()

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Query("cnn", regex="^(cnn|cnn_crf|cnn_sam|cnn_sam2)$"),
    variant: str = Query("base", regex="^(base|base_crf|grounded|grounded_crf)$"),
):
    if variant not in cnn_variants:
        return JSONResponse(status_code=400, content={"error": f"Unknown variant: {variant}"})
    cnn_model = cnn_variants[variant]

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {e}"})

    if model == "cnn":
        seg = predict_cnn_seg(cnn_model, img)
    elif model == "cnn_crf":
        seg = predict_cnn_crf_seg(cnn_model, img)
    elif model == "cnn_sam":
        seg = predict_cnn_sam_v1_seg(cnn_model, img)
    elif model == "cnn_sam2":
        seg = predict_cnn_sam_v2_seg(cnn_model, img)
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model}"})

    decision = decide_action_from_seg(seg)

    return {
        "height": int(seg.shape[0]),
        "width": int(seg.shape[1]),
        "model": model,
        "variant": variant,
        **decision,
    }