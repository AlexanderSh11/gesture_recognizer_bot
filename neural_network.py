import io

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

image_size = 64
num_classes = 29

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SimpleCNN, self).__init__()  # noqa: UP008
        
        # Используем модель ResNet
        self.backbone = models.resnet18(weights=None)
        
        # Заменяем последний слой
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class GestureRecognizer:
    def __init__(self, model_path='gesture_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        self.model = self.load_model(model_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def load_model(self, model_path):
        model = SimpleCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        return model.to(self.device)
    
    def predict(self, image_bytes):
        """Предсказание для изображения"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_value = confidence.item()
            
            print(f"Предсказание: класс {predicted_class}, уверенность {confidence_value:.4f}")
            
            return predicted_class, confidence_value
            
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка при предсказании: {e}")
            return None, 0.0

classifier = GestureRecognizer()
