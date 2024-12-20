import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.cuda.empty_cache()

def read_batch(data, df, visualize_data=False):
    # 랜덤한 항목 선택
    file_name = data[np.random.randint(len(data))]

    # ESV 및 EDV 프레임 가져오기
    esv_frame = df[df['FileName'] == file_name]['ESV'].values[0]

    # 이미지 및 마스크 경로 생성
    esv_img_path = f'{images_dir}/{file_name}/{esv_frame}.jpg'
    esv_mask_path = f'{masks_dir}/{file_name}/ESV_mask.jpg'

    # 이미지와 마스크 읽기
    img = cv2.imread(esv_img_path)
    if img is None:
        print(f"Error: Could not read image from path {esv_img_path}")
        # Handle error or return
    else:
        img = img[..., ::-1]  # Convert BGR to RGB
        
    ann_map = cv2.imread(esv_mask_path, cv2.IMREAD_GRAYSCALE)
    if ann_map is None:
        print(f"Error: Could not read mask from path {esv_mask_path}")
        # Handle error or return

    if img is None or ann_map is None:
        print(f"Error: Could not read image or mask from path {esv_img_path} or {esv_mask_path}")
        # 기본값 반환
        return np.zeros((1024, 1024, 3)), np.zeros((1, 1024, 1024)), np.zeros((1, 1, 2)), 0

    # 이미지와 마스크 크기 조정
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # 스케일링 팩터
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # 단일 바이너리 마스크 초기화
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    # 바이너리 마스크를 결합하여 단일 마스크 생성
    inds = np.unique(ann_map)[1:]  # 배경 (인덱스 0) 제외
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # 각 고유 인덱스에 대한 바이너리 마스크 생성
        binary_mask = np.maximum(binary_mask, mask)  # 기존 바이너리 마스크와 결합

    # 결합된 바이너리 마스크를 침식하여 경계 포인트 제거
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 10), np.uint8), iterations=1) #5,5

    # 침식된 마스크 내부의 모든 좌표를 가져와 무작위 포인트 선택
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        for _ in range(10):
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])

    points = np.array(points)

    if visualize_data:
        # 이미지와 포인트를 시각화
        plt.figure(figsize=(15, 5))

        # 원본 이미지
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        # 이진화된 마스크
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # 포인트가 포함된 마스크
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # 다른 색상으로 포인트 표시
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # y, x 순서로 수정

        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # 이제 shape이 (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # 이미지, 이진화된 마스크, 포인트, 마스크 수 반환
    return img, binary_mask, points, len(inds)//3 #len(inds)


data_dir = '/home/jeonk636/Capstone_DeepEcho'
images_dir = os.path.join(data_dir, "dataset/EchoNet/Images")
masks_dir = os.path.join(data_dir, "dataset/EchoNet/Masks")
fileinfo_path = os.path.join(data_dir, "dataset/EchoNet/FileInfo.csv")

fileinfo_df = pd.read_csv(fileinfo_path)

# 'Train'과 'Val'에 따라 데이터프레임 나누기
train_df = fileinfo_df[fileinfo_df['Split'] == 'Train']
test_df = fileinfo_df[fileinfo_df['Split'] == 'Val']

train_data = []
for index, row in train_df.iterrows():
    file_name = row['FileName']
    train_data.append(file_name)

# 테스트 데이터 리스트 준비 (추후 추론 또는 평가를 위해 필요할 경우)
test_data = []
for index, row in test_df.iterrows():
    file_name = row['FileName']
    test_data.append(file_name)

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {device} with mixed precision")
    with torch.autocast(device_type="cuda", dtype=torch.float16): #float16
        pass  # Replace with actual code

elif torch.backends.mps.is_available():
    device = "mps"
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
else:
    device = "cpu"
    print(f"Using device: {device}")
    

# 사전 학습된 모델 로드
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)

predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

#torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=6.5e-4,betas=(0.9, 0.996), weight_decay=3.5e-4) #1e-5, weight_decay = 4e-5

# Mix precision.
scaler = torch.cuda.amp.GradScaler()

# No. of steps to train the model.
NO_OF_STEPS = 5000 # @param 

# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "Fine_Tuned_model_esv/fine_tuned_sam2"

acuracy_loss = list()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

for step in range(1, NO_OF_STEPS + 1):
    with torch.cuda.amp.autocast():
        image, mask, input_point, num_masks = read_batch(train_data, train_df, visualize_data=False)
        if image is None or mask is None or num_masks == 0:
            continue

        # input_point은 항상 10개의 포인트를 가짐
        input_label = np.ones((input_point.shape[0], 1))  # input_point.shape[0]은 10

        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            continue

        if input_point.size == 0 or input_label.size == 0:
            continue

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue

        # 좌표와 레이블의 크기 확인 후 패딩 추가
        #print(f"패딩 전 - unnorm_coords 크기: {unnorm_coords.shape}, labels 크기: {labels.shape}")

        batch_size, num_points, coord_dims = unnorm_coords.shape
        expected_num_points = 10  # 모델이 기대하는 포인트 수로 조정

        # num_points 차원에 패딩 적용 필요 여부 확인
        if num_points != expected_num_points:
            pad_length = expected_num_points - num_points
            if pad_length > 0:
                # unnorm_coords 패딩: (좌표 차원 패딩 없음, num_points 차원 패딩)
                unnorm_coords = torch.nn.functional.pad(unnorm_coords, (0, 0, 0, pad_length), value=-1)
                # labels 패딩: num_points 차원 패딩
                labels = torch.nn.functional.pad(labels, (0, pad_length), value=-1)
            else:
                # pad_length이 음수인 경우, 잘라냄
                unnorm_coords = unnorm_coords[:, :expected_num_points, :]
                labels = labels[:, :expected_num_points]

        #print(f"패딩 후 - unnorm_coords 크기: {unnorm_coords.shape}, labels 크기: {labels.shape}")

        # 임베딩 작업 수행
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()

        # Update scheduler
        scheduler.step()

        if step % 500 == 0:
            FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
            torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

        if step == 1:
            mean_iou = 0

        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        acuracy_loss.append([mean_iou, loss.item()])

        if step % 100 == 0:
            print(f'Step {step}:\tAccuracy (IoU) = {mean_iou:.4f},\tloss = {loss.item():.4f}')
