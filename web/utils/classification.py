import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video import convert_to_imgs

classification_model_path = "model_weights/view_model.pth"

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

def get_filenames_from_directory(directory_path):
    try:
        filenames = os.listdir(directory_path)
        return filenames
    except Exception as e:
        print(f"An error occurred: {e}")

def process_videos(directory_path):
    # 모델 및 디바이스 전역 초기화

    try:
        # 1. 디렉토리에서 파일 이름 가져오기
        filenames = get_filenames_from_directory(directory_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        # 모델 불러오기
        model = CNNModel()
        model_save_path = classification_model_path
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        model.eval()  # 모델을 평가 모드로 전환합니다.

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
                    img = cv2.resize(img, (224, 224))
                    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # Convert to float
                    img = img.to(device)
                    with torch.no_grad():
                        # 모델로 예측
                        output = model(img)
                        _, predicted = torch.max(output, 1)
                        pred.append(predicted.item())

                view_files[video_path] = np.mean(pred)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

        # 3. 결과 반환
        view_files = dict(sorted(view_files.items(), key=lambda item: item[1]))
        a4c_view_files = {k: v for k, v in view_files.items() if v <= 0.1}
        if len(a4c_view_files) == 0:
            return np.array([k for k in view_files.keys()])
        return np.array([k for k in a4c_view_files.keys()])

    except Exception as e:
        print(f"An error occurred in processing videos: {e}")
        return []

def classification_a4c(directory_path):
    a4c_files = process_videos(directory_path)
    a4c_directory = os.path.join(directory_path, 'a4c')
    os.makedirs(a4c_directory, exist_ok=True)

    for file_path in a4c_files:
        try:
            file_name = os.path.basename(file_path)
            new_path = os.path.join(a4c_directory, file_name)
            os.rename(file_path, new_path)
        except Exception as e:
            print(f"Error moving file {file_path}: {e}")
    return a4c_directory