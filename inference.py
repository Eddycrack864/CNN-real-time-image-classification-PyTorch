import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

class MangoQualityCNN(nn.Module):
    def __init__(self):
        super(MangoQualityCNN, self).__init__()
        # Misma arquitectura que en el script de entrenamiento
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 clases: exportable/no exportable
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class MangoQualityDetector:
    def __init__(self, model_path):
        # Configurar dispositivo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar modelo
        self.model = MangoQualityCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_quality(self, frame):
        # Convertir frame de OpenCV a PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Redimensionar a 224x224 si es necesario
        pil_image = pil_image.resize((224, 224))
        
        # Aplicar transformaciones
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        # Mapear predicción a etiqueta
        labels = ['No Exportable', 'Exportable']
        confidence = probabilities[0][predicted.item()].item()
        
        return labels[predicted.item()], confidence
    
    def run_real_time_detection(self):
        # Abrir cámara
        cap = cv2.VideoCapture(0)
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detección de calidad
            quality, confidence = self.detect_quality(frame)
            
            # Color y texto basado en la calidad
            color = (0, 255, 0) if quality == 'Exportable' else (0, 0, 255)
            
            # Mostrar resultado
            cv2.putText(frame, 
                        f'{quality} (Confianza: {confidence:.2f})', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        color, 
                        2)
            
            cv2.imshow('Detección de Calidad de Mango', frame)
            
            # Salir con tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

# Ejecutar detección en tiempo real
if __name__ == '__main__':
    detector = MangoQualityDetector('mejor_modelo_mangos.pth')
    detector.run_real_time_detection()