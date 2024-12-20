import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
from web.utils.video import *
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, classification_report


np.random.seed(42)

hospital_path = "/home/behan/Capstone_DeepEcho/dataset/hospital_31"
a4c_path = "/home/behan/Capstone_DeepEcho/sandbox/behan/a4c_names.csv"
save_dir = "/home/behan/Capstone_DeepEcho/dataset/Classification_Dataset"
rvenet_names_path = "/home/behan/Capstone_DeepEcho/sandbox/behan/RVENet_names.csv"
rvenet_path = "/home/behan/Capstone_DeepEcho/dataset/RVENet/Videos"
data_dir = "/home/behan/Capstone_DeepEcho/dataset/Classification_Dataset"
model_save_path = "/home/behan/Capstone_DeepEcho/sandbox/behan/trained_models/view_model.pth"
train_log_save_path = 'view_training_log.csv'

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Layer 1: Convolutional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Layer 2: Convolutional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # Layer 3: MaxPooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4: Convolutional
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Layer 5: Convolutional
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Layer 6: MaxPooling
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 7: Convolutional
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Layer 8: Convolutional
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # Layer 10: MaxPooling
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 19: Flatten
        # Flattening is handled in the forward method using `view`.

        # Layer 20: Fully Connected Layer
        self.fc1 = nn.Linear(128 * 28 * 28, 1028)  # Assuming input image size is (224, 224)
        # Layer 21: Fully Connected Layer
        self.fc2 = nn.Linear(1028, 512)
        # Layer 22: Softmax Layer
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully connected layers + ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer with Softmax
        # x = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)

        return x

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.classes = os.listdir(image_folder)
        self.classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for cls in self.classes:
            cls_folder = os.path.join(image_folder, cls)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        # 이미지를 열고 그레이스케일로 변환
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        # Albumentations 변환 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def get_filenames_from_directory(directory_path):
    try:
        filenames = os.listdir(directory_path)
        return filenames
    except Exception as e:
        print(f"An error occurred: {e}")
        
def save_oth_video_to_imgs(filenames, save_dir, i = 0):
    os.makedirs(f"{save_dir}", exist_ok=True)
    for f_name in tqdm(filenames):
        names = get_filenames_from_directory(f"{hospital_path}/{f_name}")
        names = np.array(names)
        names = [name for name in names if name.endswith('.mp4')]
        for n in names:
            f_n = n.split(".")[0]
            if f_n in a4c_names:
                continue
            imgs, fps, width, height = convert_to_imgs(f"{hospital_path}/{f_name}/{n}")
            for j, img in enumerate(imgs):
                cv2.imwrite(f"{save_dir}/{i}.jpg", img)
                i += 1

def save_oth_a4c_video_to_imgs(filenames, save_dir, i = 0):
    os.makedirs(f"{save_dir}", exist_ok=True)
    for f_name in tqdm(filenames):
        names = get_filenames_from_directory(f"{hospital_path}/{f_name}")
        names = np.array(names)
        names = [name for name in names if name.endswith('.mp4')]
        for n in names:
            f_n = n.split(".")[0]
            if f_n not in a4c_names:
                continue
            imgs, fps, width, height = convert_to_imgs(f"{hospital_path}/{f_name}/{n}")
            for j, img in enumerate(imgs):
                cv2.imwrite(f"{save_dir}/{i}.jpg", img)
                i += 1

def save_oth_train_val_imgs(filenames, save_dir, k, n): # 0 < n < k
    val_cnt = len(filenames) // k
    val_filenames = filenames[val_cnt*n:val_cnt*(n+1)]
    train_filenames = np.setdiff1d(filenames, val_filenames)
    save_oth_video_to_imgs(train_filenames, f"{save_dir}/train/Others")
    save_oth_video_to_imgs(val_filenames, f"{save_dir}/valid/Others")

def remove_train_val_imgs(save_dir):
    os.system(f"rm -rf {save_dir}/train")
    os.system(f"rm -rf {save_dir}/valid")
     
def save_rve_video_to_imgs(filenames, save_dir, i = 0):
    os.makedirs(save_dir, exist_ok=True)
    for f_name in tqdm(filenames):
        imgs, fps, width, height = convert_to_imgs(f"{rvenet_path}/{f_name}.avi")
        for j, img in enumerate(imgs):
            cv2.imwrite(f"{save_dir}/{i}.jpg", img)
            i += 1

def save_rve_train_val_imgs(filenames, save_dir, k, n): # 0 < n < k
    val_cnt = len(filenames) // k
    val_filenames = filenames[val_cnt*n:val_cnt*(n+1)]
    train_filenames = np.setdiff1d(filenames, val_filenames)
    save_rve_video_to_imgs(train_filenames, f"{save_dir}/train/A4C")
    i = len(get_filenames_from_directory(f"{save_dir}/train/A4C"))
    save_oth_video_to_imgs(a4c_names, f"{save_dir}/train/A4C", i)
    save_rve_video_to_imgs(val_filenames, f"{save_dir}/valid/A4C")

a4c_names = pd.read_csv(a4c_path)
a4c_names = a4c_names["FileName"].values
a4c_names = np.array(a4c_names)

others_filenames = get_filenames_from_directory(hospital_path)
others_train_filenames = np.random.choice(others_filenames, 26, replace=False)
others_test_filenames = np.setdiff1d(others_filenames, others_train_filenames)

rvenet_names = pd.read_csv(rvenet_names_path)
rvenet_names = rvenet_names["FileName"].values
rvenet_names = np.array(rvenet_names)


rvenet_train_filenames = np.random.choice(rvenet_names, int(len(rvenet_names)*0.8), replace=False)
revnet_test_filenames = np.setdiff1d(rvenet_names, rvenet_train_filenames)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(), 
    # transforms.Lambda(lambda x: x * 0.1),  # [0, 1] 범위를 [0, 0.1]로 스케일링
    # transforms.Normalize(mean=mean, std=std)  # 정규화 (선택 사항)
])

batch_size = 64


model = CNNModel()
model.to(device)

# 증강 파이프라인 정의
aug_transform = A.Compose([
    A.Rotate(limit=10, p=0.5, border_mode=0),  # 최대 10도 회전
    A.ShiftScaleRotate(
        shift_limit=0.1,       # 가로 및 세로 이동 최대 10%
        scale_limit=0.08,      # 최대 8% 줌
        rotate_limit=0,        # 추가적인 회전은 없음
        p=0.5,
        border_mode=0
    ),
    A.Affine(
        shear=(-1.718, 1.718),  # 최대 0.03 라디안 전단 (약 ±1.718도)
        p=0.5,
        fit_output=True,
        mode=0  # 상수 패딩 모드
    ),
    A.HorizontalFlip(p=0.5),  # 수평 뒤집기
    A.VerticalFlip(p=0.5),    # 수직 뒤집기
    A.Resize(224, 224),       # 이미지를 목표 크기로 리사이즈
    # A.Normalize(mean=mean, std=std),  # 계산된 평균과 표준 편차로 정규화
    ToTensorV2(),             # PyTorch 텐서로 변환
])


    
oth_train_names = others_train_filenames
rve_train_names = rvenet_train_filenames
remove_train_val_imgs(save_dir)
oth_val_cnt = len(oth_train_names) // 5
rve_val_cnt = len(rve_train_names) // 5
oth_val_names = np.random.choice(oth_train_names, oth_val_cnt, replace=False)
rve_val_names = np.random.choice(rve_train_names, rve_val_cnt, replace=False)
oth_train_names = np.setdiff1d(oth_train_names, oth_val_names)
rve_train_names = np.setdiff1d(rve_train_names, rve_val_names)
print("---------------start save imgs----------------")
save_rve_video_to_imgs(rve_train_names, f"{save_dir}/train/A4C")
i = len(get_filenames_from_directory(f"{save_dir}/train/A4C"))
save_oth_a4c_video_to_imgs(oth_train_names, f"{save_dir}/train/A4C", i)
save_rve_video_to_imgs(rve_val_names, f"{save_dir}/valid/A4C")

for _ in tqdm(range(2)):
    train_aug_dataset = CustomImageDataset(image_folder=f"{save_dir}/train", transform=aug_transform)
    train_aug_loader = DataLoader(train_aug_dataset, shuffle=False)
    i = len(get_filenames_from_directory(f"{save_dir}/train/A4C"))
    for img, label in train_aug_loader:
        cv2.imwrite(f"{save_dir}/train/A4C/{i}.jpg", img[0].squeeze().numpy())
        i += 1
    val_aug_dataset = CustomImageDataset(image_folder=f"{save_dir}/valid", transform=aug_transform)
    val_aug_loader = DataLoader(val_aug_dataset, shuffle=False)
    j = len(get_filenames_from_directory(f"{save_dir}/valid/A4C"))
    for img, label in val_aug_loader:
        cv2.imwrite(f"{save_dir}/valid/A4C/{j}.jpg", img[0].squeeze().numpy())
        j += 1
save_oth_video_to_imgs(oth_train_names, f"{save_dir}/train/Others")
save_oth_video_to_imgs(oth_val_names, f"{save_dir}/valid/Others")
print("---------------save imgs done----------------")


# Compute confusion matrix
def calc_matrix(all_labels, all_predictions):
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    # Compute Sensitivity, Specificity, and F1-Score
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = sensitivity  # Same as sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)
    return sensitivity, specificity, f1_score

batch_size = 64

# 옵티마이저 설정
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 손실 함수 설정 (categorical crossentropy와 유사한 CrossEntropyLoss 사용)
criterion = nn.CrossEntropyLoss()

# k = 5

epochs = 5
for epoch in range(epochs):
    # n = epoch % k
    # save_k_fold(others_train_filenames, rvenet_train_filenames, save_dir, k, n)
        
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    val_data = datasets.ImageFolder(root=f"{data_dir}/valid", transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    
    # Model Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_all_labels = []
    train_all_predictions = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파 및 옵티마이저 업데이트
        loss.backward()
        optimizer.step()

        # 통계 수집
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_all_labels.extend(labels.cpu().numpy())
        train_all_predictions.extend(predicted.cpu().numpy())

    # 에포크당 정확도 및 손실 출력
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    
    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_all_labels = []
    val_all_predictions = []

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

            val_all_labels.extend(val_labels.cpu().numpy())
            val_all_predictions.extend(val_predicted.cpu().numpy())

    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_accuracy = val_correct / val_total
    
    train_sensitive, train_specificity, train_f1 = calc_matrix(train_all_labels, train_all_predictions)
    val_sensitive, val_specificity, val_f1 = calc_matrix(val_all_labels, val_all_predictions)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Sensitivity: {train_sensitive:.4f}, Specificity: {train_specificity:.4f}, F1-Score: {train_f1:.4f}")
    print(f"\tValidation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}, Sensitivity: {val_sensitive:.4f}, Specificity: {val_specificity:.4f}, F1-Score: {val_f1:.4f}")
    # CSV 파일에 기록할 데이터
    csv_data = {
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_accuracy,
        'train_sensitivity': train_sensitive,
        'train_specificity': train_specificity,
        'train_f1': train_f1,
        'val_loss': val_epoch_loss,
        'val_accuracy': val_epoch_accuracy,
        'val_sensitivity': val_sensitive,
        'val_specificity': val_specificity,
        'val_f1': val_f1
    }

    # CSV 파일에 기록
    csv_file = train_log_save_path
    csv_columns = list(csv_data.keys())

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if csvfile.tell() == 0:  # 파일이 비어있으면 헤더 작성
                writer.writeheader()
            writer.writerow(csv_data)
    except IOError:
        print("I/O error")
print("Training Finished.")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")