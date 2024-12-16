import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from PIL import Image

# Filtro personalizado para imágenes
class ImageSizeFilterDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_size=(224, 224)):
        super().__init__(root, transform)
        self.target_size = target_size
        self.filtered_samples = self._filter_images()

    def _filter_images(self):
        filtered_samples = []
        for path, label in self.samples:
            try:
                with Image.open(path) as img:
                    # Convertir a RGB si es necesario
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Verificar y redimensionar
                    if img.size != self.target_size:
                        img = img.resize(self.target_size, Image.LANCZOS)
                        img.save(path)  # Sobrescribir imagen original
                    
                    filtered_samples.append((path, label))
            except Exception as e:
                print(f"Error procesando {path}: {e}")
        return filtered_samples

    def __getitem__(self, index):
        path, label = self.filtered_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.filtered_samples)

def count_images_in_dataset(data_dir):
    """Contar imágenes en cada subdirectorio"""
    image_counts = {}
    for subset in ['train', 'val']:
        subset_path = os.path.join(data_dir, subset)
        if os.path.exists(subset_path):
            subset_counts = {}
            for classe in os.listdir(subset_path):
                classe_path = os.path.join(subset_path, classe)
                if os.path.isdir(classe_path):
                    count = len([f for f in os.listdir(classe_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    subset_counts[classe] = count
            image_counts[subset] = subset_counts
    return image_counts

def train_model(data_dir, num_epochs=20, batch_size=32, learning_rate=0.001):
    # Contar y mostrar imágenes
    image_counts = count_images_in_dataset(data_dir)
    print("\n📸 Conteo de Imágenes:")
    for subset, classes in image_counts.items():
        print(f"\n{subset.upper()}:")
        for classe, count in classes.items():
            print(f"  {classe}: {count} imágenes")

    # Configuración de transformaciones
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # Cargar datasets con filtro de tamaño
    image_datasets = {
        x: ImageSizeFilterDataset(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Crear dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

class MangoQualityCNN(nn.Module):
    def __init__(self):
        super(MangoQualityCNN, self).__init__()
        # Modificamos la arquitectura para 224x224
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

def train_model(data_dir, num_epochs=20, batch_size=32, learning_rate=0.001):
    # Configuración de transformaciones
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # Cargar datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Crear dataloaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # Configuración del dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Inicializar modelo
    model = MangoQualityCNN().to(device)

    # Configuración de entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        
        # Entrenar
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calcular precisiones
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        print(f"Precisión de entrenamiento: {train_accuracy:.2f}%")
        print(f"Precisión de validación: {val_accuracy:.2f}%")
        
        # Guardar el mejor modelo
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), '/content/drive/MyDrive/dataset/modelo_mangos.pth')
            print("Modelo guardado!")

    print("\nEntrenamiento completado.")
    return model

# Ejecutar entrenamiento
if __name__ == '__main__':
    data_directory = '/content/drive/MyDrive/dataset'  # Ajusta esta ruta según tu estructura
    train_model(data_directory)