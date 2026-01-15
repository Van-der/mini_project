import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from augmentdatting import get_transforms
from train_model import DualBranchDeepfakeDetector

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Fix: Use full backward hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Enable gradients for input
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        logit = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logit.argmax().item()
        
        # Get scalar score for target class (keep in graph!)
        score = logit[0, class_idx]
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def predict_image(image_path):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. FACE DETECTION (production quality)
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device=DEVICE)  # Single face
    
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    
    # Detect & crop
    face_boxes, _ = mtcnn.detect(img_np)
    if face_boxes is None or len(face_boxes) == 0:
        print("❌ No face detected - using full image")
        img_cropped = cv2.resize(img_np, (224, 224))
    else:
        print(f"✅ Face detected: {len(face_boxes)} face(s)")
        # Extract first face
        box = face_boxes[0]
        x1, y1, x2, y2 = box.astype(int)
        img_cropped = img_np[y1:y2, x1:x2]
        img_cropped = cv2.resize(img_cropped, (224, 224))
    
    # 2. Normalize
    transform = get_transforms(train=False)
    img_tensor = transform(image=img_cropped)['image'].unsqueeze(0).to(DEVICE)
    
    # Rest unchanged...

    
    # Model (rest unchanged...)
    model = DualBranchDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    img_tensor.requires_grad_(True) 
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax().item()
        confidence = probs.max().item()
    
    classes = ['Real', 'Deepfake', 'AI-Gen']
    print(f"Prediction: {classes[pred_idx]} ({confidence:.1%})")
    
    # Grad-CAM & viz
    # Get last conv layer from EfficientNet backbone
    target_layer = model.rgb_backbone._conv_head
    gradcam = GradCAM(model, target_layer)
 
    cam = gradcam.generate(img_tensor, pred_idx)
    
    img_np = np.array(img_pil)
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(img_np); plt.title('Input'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(cam_resized); plt.title('Grad-CAM'); plt.axis('off')
    overlay = 0.4*img_np + 0.6*cam_resized[:,:,None]*255
    plt.subplot(1,3,3); plt.imshow(overlay.astype(np.uint8)); plt.title(f'{classes[pred_idx]} ({confidence:.1%})'); plt.axis('off')
    plt.savefig('gradcam_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: gradcam_result.png")


if __name__ == '__main__':
    import sys
    predict_image(sys.argv[1])
