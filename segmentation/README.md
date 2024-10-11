# 실행 전 주의
> git에는 train된 unet모델이 없습니다.
> trained_models 디렉토리에 train된 U-Net 모델을 먼저 넣어주세요.
> unet 모델 파일의 이름을 "unet.pth"로 설정하면 작동가능합니다.
> unet 모델 파일의 이름을 다른 것으로 설정할 경우, **segmentation_viedo.py** 에서 "unet_path" 변수를 알맞게 변경해주세요.

# 실행 방법
1. 기본: dataset/VideoPredictor 디렉토리에 동영상 파일 저장됨.
```
python3 ./segmentation_video.py <video_path>
```
2. 저장 디렉토리 지정
```
python3 ./segmentation_video.py <video_path> -o <output_path>
```