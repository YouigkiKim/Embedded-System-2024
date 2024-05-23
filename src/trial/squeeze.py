import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class SqueezeNetRegressor(nn.Module):
    def __init__(self, pretrained=False):
        super(SqueezeNetRegressor, self).__init__()
        self.squeezenet = models.squeezenet1_0(pretrained=pretrained)
        self.squeezenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))  # 하나의 출력값
        self.squeezenet.num_classes = 1

        if not pretrained:
            self._initialize_weights()

    def forward(self, x):
        x = self.squeezenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # 크기를 1x1로 줄임
        x = self.squeezenet.classifier(x).view(x.size(0), -1)  # 하나의 좌표 (x)를 출력
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, label_file, transform=None):
        self.labels = self._load_labels(label_file)
        self.transform = transform

    def _load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = {}
            for line in f:
                parts = line.strip().split()
                image_path = parts[0]
                x_coord = float(parts[1])
                labels[image_path] = x_coord
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = list(self.labels.keys())[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[image_path]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# 데이터 전처리 설정
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((640, 540)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((640, 540)),
        transforms.ToTensor()
    ]),
}

# 데이터셋 및 데이터 로더 설정
label_file = 'dataset/common/annotation.txt'  # annotation 파일의 경로를 지정하세요.
batch_size = 8

train_dataset = CustomDataset(label_file, transform=data_transforms['train'])
val_dataset = CustomDataset(label_file, transform=data_transforms['val'])  # 필요에 따라 검증 데이터셋을 다르게 설정할 수 있습니다.
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

# 모델, 손실 함수 및 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = SqueezeNetRegressor(pretrained=False).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')

    return model

# 모델 학습
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

# 예측 및 변환 함수 정의 (앞서 설명한 것과 동일)
def denormalize_x(x_normalized, image_width=640):
    x_pixel = int(x_normalized * image_width)
    return x_pixel

def predict_and_convert(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        x_normalized = output[0].item()
    
    x_pixel = denormalize_x(x_normalized)
    return x_pixel

# 예제 사용법
image_path = 'dataset/common/framge_000000000.jpg'
x_pixel = predict_and_convert(model, image_path, data_transforms['val'], device)
print(f'Predicted x-coordinate in pixels: {x_pixel}')
