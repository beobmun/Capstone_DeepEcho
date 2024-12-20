import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from train_model import *


model_save_path = "/home/behan/Capstone_DeepEcho/sandbox/behan/trained_models/view_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hospital_path = "/home/behan/Capstone_DeepEcho/dataset/hospital_31"

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
    
def process_videos(directory_path, test_mode=False):
    # 모델 및 디바이스 전역 초기화

    try:
        # 1. 디렉토리에서 파일 이름 가져오기
        filenames = get_filenames_from_directory(directory_path)

        # 모델 불러오기
        model = CNNModel()
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        model.eval()  # 모델을 평가 모드로 전환합니다.
        # print("Model loaded successfully.")

        # 2. 동영상 파일 필터링 및 뷰 분류
        view_files = dict()
        for filename in filenames:
            if not filename.endswith('.mp4'):
                continue
            video_path = os.path.join(directory_path, filename)

            # test_video로 결과 예측
            try:
                imgs, fps, width, height = convert_to_imgs(video_path)
                
                # 이미지들로 모델 예측
                pred = []
                for img in imgs:
                    with torch.no_grad():
                        img = cv2.resize(img, (224, 224))
                        img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # Convert to float
                        
                        img = img.to(device)

                        # 모델로 예측
                        output = model(img)
                        _, predicted = torch.max(output, 1)
                        pred.append(predicted.item())

                    # A4C 뷰(0번 클래스)인 경우 처리
                # print(f"{os.path.basename(video_path)}:, {np.mean(pred)}")
                # if np.mean(pred) < 0.1:
                view_files[video_path] = np.mean(pred)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        # 3. 결과 반환
        view_files = dict(sorted(view_files.items(), key=lambda item: item[1]))
        a4c_view_files = {k: v for k, v in view_files.items() if v <= 0.1}
        if len(a4c_view_files) == 0 or test_mode:
            return view_files
        return a4c_view_files

    except Exception as e:
        print(f"An error occurred in processing videos: {e}")
        return []


clf_results = []
for test_file_name in others_test_filenames:
    h_path = f"{hospital_path}/{test_file_name}"
    a4c = process_videos(h_path, test_mode=True)
    clf_results.append(a4c)

results = dict()
score = []
true_label = []
for a4c in clf_results:
    for k, v in a4c.items():
        n = os.path.basename(k).split(".")[0]
        if n in a4c_names:
            true_label.append(1)
            t = 1
        else:
            true_label.append(0)
            t = 0
        score.append(v)
        s = v
        results[k] = [t, s]
        
        from sklearn.metrics import roc_curve, auc

# Calculate FPR, TPR, and AUC
fpr, tpr, th = roc_curve(true_label, score, pos_label=0)
roc_auc = auc(fpr, tpr)

# Plot AUC-ROC curve
plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot(fpr, tpr, color='#439f97', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
print(th)