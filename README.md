# Keras-Object Detector

Keras 기반의 classifier, object detector 구현 코드입니다.  
아래 각 디렉토리 별 설명을 참고해주세요.

```
Project
├─ .gitignore
├─ Classifier : Classification 모델의 학습 코드 및 Generator 코드가 정의돼 있습니다.
├─ Detector : Detection 모델의 학습 코드 및 Generator 코드가 정의돼 있습니다.
├─ map : Detector의 성능 지표인 mAP(mean average precision) 계산을 위한 모듈입니다.
├─ models : 모델의 구성을 정의한 디렉토리입니다.
│  ├─ Backbones.py : 기본 Backbone 구현
│  ├─ Head.py : Detection Head 구현
│  ├─ Layers.py : 공통으로 쓰는 Convolution Block 구현
│  └─ LossFunc.py : Detector Loss 함수(Focal Loss, cIoU Loss 등) 구현
├─ Scripts : Anchor Cluster, Deploy를 위한 freeze 등 기타 유틸 스크립트
├─ Utils : 학습 시 사용하는 Callback 등 유틸 함수 구현
└─ Datasets : 학습 데이터 파일이 위치하는 경로입니다. BDD Dataset을 테스트 했던 스크립트만 추가돼있습니다.

```

### Reference

기본적인 Detector의 구성은 [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) 를 기본으로 한 One-Stage Detector 입니다.  
정확도, 성능 향상을 위해 FPN, Focal Loss, cIoU Loss, Anchor Clustering 등 여러 기법을 참고하여 구현했습니다.

구현된 코드들은 다음 논문 및 오픈소스를 참고했습니다.

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  
  [models/LossFunc.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/LossFunc.py#L119)  
  PyTorch, Keras 로 구현된 오픈소스를 참고했습니다.

- [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)  
  [models/LossFunc.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/LossFunc.py#L34)  
  공식 PyTorch cIoU 코드를 참고하여 Keras 로 구현했습니다.

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)  
  [models/Backbones.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/Backbones.py#L8)

- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)  
  [models/Backbones.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/Backbones.py#L52)

- [Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units](https://arxiv.org/abs/1603.05201)  
  [models/Layers.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/Layers.py#L272)

- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)  
  [models/Layers.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/Layers.py#L7)  
  논문과 다소 다르게 구현이 되어있습니다. ex) FC -> Conv, Bias Add -> BatchNorm  
  해당 부분은 NPU 포팅 시 Quantization 관련 이슈가 발생하여 비슷한 연산으로 변경하여 구현했습니다.

- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)  
  [models/Layers.py](https://github.com/nalnez13/Keras-Object-Detector/blob/22c4fdf03d064f544e35f047db3a5c382f66bd7c/models/Layers.py#L239)

기타 [CSPNet](https://arxiv.org/abs/1911.11929), [Partial Residual Block](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LPCV/Wang_Enriching_Variety_of_Layer-Wise_Learning_Information_by_Gradient_Combination_ICCVW_2019_paper.pdf), [Dynamic Convolution](https://arxiv.org/abs/1912.03458) 을 적용했으며, 정확도 향상이 미미하거나 Keras 구조상 제대로 구현이 되지 않은 코드도 일부 존재합니다.
