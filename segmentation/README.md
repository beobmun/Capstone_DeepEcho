# 실행 전 주의
> git에는 train된 unet모델이 없습니다.

> trained_models 디렉토리에 train된 U-Net 모델을 먼저 넣어주세요.

> unet 모델 파일의 이름을 "unet.pth"로 설정하면 작동가능합니다.

> unet 모델 파일의 이름을 다른 것으로 설정할 경우, **segmentation_viedo.py** 에서 "unet_path" 변수를 알맞게 변경해주세요.

# 실행 방법
1. 기본: ../dataset/VideoPredictor 디렉토리에 동영상 파일 저장됩니다.
```
python3 ./segmentation_video.py <video_path>
```
2. 저장 디렉토리 지정이 가능합니다.
```
python3 ./segmentation_video.py <video_path> -o <output_path>
```
3. 여러 영상을 연속으로 segmentation 가능합니다. 그러나 GPU 메모리 부족으로 인한 에러가 발생할 수 있습니다.**(사용 비추천)**
```
python3 ./segmentation_video.py <video_path_1> <video_path_2> ...

    or

python3 ./segmentation_video.py <video_path_1> <video_path_2> ... -o <output_path>
```
