import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Load Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Function to load and process image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB for visualization
    return image

# Function for Grad-CAM (simple version)
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad = True  # Ensure gradients
    return img_tensor

# Get the last convolutional layer for Grad-CAM
target_layer = model.backbone.body.layer4[-1]

# Hook functions to extract gradients and activations
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# Register hooks
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Compute Grad-CAM
def compute_gradcam(img_tensor, model, img_size):
    gradients.clear()
    activations.clear()

    model.zero_grad()
    output = model(img_tensor)

    # Get the class prediction and backpropagate
    class_idx = output[0]['scores'].argmax().item()
    output[0]['scores'][class_idx].backward(retain_graph=True)

    grads = gradients[0].cpu().detach().numpy()
    acts = activations[0].cpu().detach().numpy()

    weights = np.mean(grads, axis=(2, 3), keepdims=True)
    gradcam = np.sum(weights * acts, axis=1)[0]
    gradcam = np.maximum(gradcam, 0)  # ReLU
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)  # Normalize

    # Resize to match original image size
    gradcam = cv2.resize(gradcam, img_size)

    return gradcam

# Function to create a breast mask
def create_breast_mask(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)  # Smooth edges

    mask = mask.astype(np.float32) / 255.0
    return mask

# Overlay Grad-CAM heatmap onto image
def overlay_heatmap(img, heatmap, mask, alpha=0.6):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Suppress heatmap outside breast area
    heatmap = (heatmap * mask[:, :, np.newaxis]).astype(np.uint8)

    overlay = Image.blend(img, Image.fromarray(heatmap), alpha)
    return overlay

# Function to predict and draw bounding boxes
def predict_and_draw_boxes(image_path, score_threshold=0.2, tissue_threshold=200, box_margin=0.1, return_boxes=False):
    image = load_image(image_path)
    original_width, original_height = image.size

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    masks = predictions[0]['masks']

    draw = ImageDraw.Draw(image)

    detected_areas = []
    boxes_list = []

    for i in range(len(boxes)):
        if scores[i] > score_threshold:
            box = boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box

            mask = masks[i, 0]
            mask = mask > 0.5
            mask = mask.cpu().numpy()

            image_array = np.array(image)
            tissue_area = image_array > tissue_threshold
            # Ensure mask has 3 channels
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            refined_mask = np.logical_and(mask, tissue_area)

            rows = np.any(refined_mask, axis=1)
            cols = np.any(refined_mask, axis=0)

            if np.any(rows) and np.any(cols):
                y1_new, y2_new = np.where(rows)[0][[0, -1]]
                x1_new, x2_new = np.where(cols)[0][[0, -1]]

                box_width = x2_new - x1_new
                box_height = y2_new - y1_new

                margin_x = int(box_width * box_margin)
                margin_y = int(box_height * box_margin)

                x1_new += margin_x
                y1_new += margin_y
                x2_new -= margin_x
                y2_new -= margin_y

                x1_new = max(x1_new, 0)
                y1_new = max(y1_new, 0)
                x2_new = min(x2_new, original_width)
                y2_new = min(y2_new, original_height)

                overlap = False
                for area in detected_areas:
                    ix1, iy1, ix2, iy2 = area
                    if not (x2_new < ix1 or x1_new > ix2 or y2_new < iy1 or y1_new > iy2):
                        overlap = True
                        break

                if not overlap:
                    detected_areas.append([x1_new, y1_new, x2_new, y2_new])
                    boxes_list.append([x1_new, y1_new, x2_new, y2_new])

                    # Drawing the box with dark blue color and increased thickness
                    draw.rectangle([x1_new, y1_new, x2_new, y2_new], outline="darkblue", width=8)

    if return_boxes:
        return boxes_list

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Final function to process image with Grad-CAM and bounding boxes
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_size = (image.width, image.height)

    # Compute Grad-CAM
    full_img_tensor = preprocess_image(image)
    full_heatmap = compute_gradcam(full_img_tensor, model, img_size)

    # Normalize Grad-CAM globally
    full_heatmap = (full_heatmap - full_heatmap.min()) / (full_heatmap.max() - full_heatmap.min() + 1e-8)

    # Generate a mask for breast region
    breast_mask = create_breast_mask(image)

    # Apply Grad-CAM with mask to full image
    full_overlay = overlay_heatmap(image, full_heatmap, breast_mask, alpha=0.5)

    # Run detection to get bounding boxes
    boxes = predict_and_draw_boxes(image_path, return_boxes=True)

    # Draw bounding boxes on final overlay
    draw = ImageDraw.Draw(full_overlay)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

    # Increase contrast for better visibility
    enhancer = ImageEnhance.Contrast(full_overlay)
    final_image = enhancer.enhance(1.2)

    # Save or show final image with Grad-CAM and bounding boxes
    final_image_path = image_path.replace(".png", "_result.png")
    final_image.save(final_image_path)
    return final_image_path