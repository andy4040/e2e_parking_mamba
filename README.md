## 1. 프로젝트 요약
🚗 Mamba 기반 End-to-End 자율 주차 시스템
CARLA 시뮬레이터 환경에서 Mamba(State Space Model) 아키텍처를 활용하여 구현한 효율적인 End-to-End 자율 주차 에이전트입니다.

본 프로젝트는 기존의 Transformer 구조보다 연산 효율이 뛰어난 **Mamba(Selective State Space Model)**를 활용하여 주행 제어 명령을 직접 생성합니다. 복잡한 주차 시나리오에서 센서 데이터를 입력받아 조향(Steering), 가속(Acceleration), 제동(Brake) 값을 실시간으로 예측합니다.

본 모델은 다중 센서 정보를 통합하여 정밀한 주차 제어를 수행하기 위해 다음과 같은 단계별 구조를 가집니다.

1. Multi-View Input (4 Surround Images): 차량 주변을 감시하는 4개의 카메라(전/후/좌/우)로부터 원시 데이터를 입력받습니다.

2. BEV Feature Extraction: 각 카메라의 2D 이미지를 공간 지각력이 극대화된 BEV(Bird's-Eye View) 특징 맵으로 변환합니다. 이는 주차 칸과의 거리 및 정렬 상태를 파악하는 데 핵심적인 역할을 합니다.

3. Feature Fusion: 추출된 각 뷰의 특징들을 하나의 통합된 벡터 공간으로 융합(Fusion)하여 차량 주변 360도 환경을 단일 지표로 재구성합니다.

4. Mamba Sequence Modeling: 융합된 특징 시퀀스를 Mamba(SSM) 블록에 입력합니다. Mamba는 기존 Transformer 대비 압도적인 연산 효율로 과거 주행 이력을 참조하여 현재의 주행 맥락을 파악합니다.

5. Control Prediction: 최종적으로 Mamba의 출력값은 제어 헤드를 거쳐 조향(Steering), 가속(Accel), 제동(Brake) 등의 물리적 제어 신호로 변환됩니다.

30epoch 결과 (loss:0.437 / val_loss:0.81)
## 2. Requirement
carla: 0.9.11

python: 3.7


## 3. 결과

NVIDIA RTX 2080 GPU 환경에서 총 30 Epoch 동안 약 60시간의 학습을 진행하였다.

🏁 주차 성공 판정 기준 

X축 이격 거리: ± 1.0 m — 주차 칸 진입 깊이(전/후 방향)의 허용 오차

Y축 이격 거리: ± 0.6 m — 주차선 사이 중앙 정렬(좌/우 방향)의 허용 오차

방향 오차: ± 10° — 주차 칸과 차량의 수평 정렬 및 평행 상태의 허용 오차


<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/ce9144c7-e88b-4b79-9458-55c2d8440b30" /><img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/b4655293-d00f-4a7c-ac20-5e203925b05b" />


## 4. Dataset

https://pan.baidu.com/s/1PoMSfgZQMnUGlhi7S5fFZw?pwd=2ik6

pretrained model

last.ckpt 구글 드라이브 다운로드


## 5. References

본 프로젝트는 아래 논문의 연구 성과를 바탕으로 구현 및 개선되었습니다.

[E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](https://ieeexplore.ieee.org/document/10588551)



