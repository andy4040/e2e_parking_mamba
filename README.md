## 1. 프로젝트 요약
🚗 Mamba 기반 End-to-End 자율 주차 시스템
CARLA 시뮬레이터 환경에서 Mamba(State Space Model) 아키텍처를 활용하여 구현한 효율적인 End-to-End 자율 주차 에이전트입니다.

본 프로젝트는 기존의 Transformer 구조보다 연산 효율이 뛰어난 Mamba(Selective State Space Model)를 활용하여 주행 제어 명령을 직접 생성합니다. 복잡한 주차 시나리오에서 센서 데이터를 입력받아 조향(Steering), 가속(Acceleration), 제동(Brake) 값을 실시간으로 예측합니다.

본 모델은 다중 센서 정보를 통합하여 정밀한 주차 제어를 수행하기 위해 다음과 같은 단계별 구조를 가집니다.

1. Multi-View Input (4 Surround Images): 차량 주변을 감시하는 4개의 카메라(전/후/좌/우)로부터 원시 데이터를 입력받습니다.

2. BEV Feature Extraction: 각 카메라의 2D 이미지를 공간 지각력이 극대화된 BEV(Bird's-Eye View) 특징 맵으로 변환합니다. 이는 주차 칸과의 거리 및 정렬 상태를 파악하는 데 핵심적인 역할을 합니다.

3. Feature Fusion: 추출된 각 뷰의 특징들을 하나의 통합된 벡터 공간으로 융합(Fusion)하여 차량 주변 360도 환경을 단일 지표로 재구성합니다.

### 4. Mamba Sequence Modeling (핵심 메커니즘)

트랜스포머가 융합한 특징 시퀀스를 **Mamba(SSM) 블록**에 입력한다. 이 과정에서 **병렬 스캔(Parallel Scan)** 기법을 사용하여 시퀀스 길이에 대해 O(log L)의 복잡도로 고속 병렬 처리를 수행한다.

특히 S6 설계 시 입력(`B`)과 출력(`C`) 가중치를 공유하는 **simplified S6 구조**를 적용하여 파라미터를 효율화했다. 이는 학습 속도 향상뿐만 아니라, 주행 맥락을 제어 신호로 변환할 때의 **일관성과 안정성**을 극대화하는 핵심 역할을 한다.

5. Control Prediction: 최종적으로 Mamba의 출력값은 제어 헤드를 거쳐 조향(Steering), 가속(Accel), 제동(Brake) 등의 물리적 제어 신호로 변환됩니다.

<img width="1070" height="434" alt="image" src="https://github.com/user-attachments/assets/edd4270a-0489-495e-8905-191d364b240f" />



## 2. Requirement
carla: 0.9.11

python: 3.7


## 3. 결과

NVIDIA RTX 2080 GPU 환경에서 총 30 Epoch 동안 약 60시간의 학습을 진행하였다.

🏁 주차 성공 판정 기준 


차량의 중심이 주차 칸의 정중앙으로부터 0.5m 이내에 위치해야 하며, 차체가 목표 방향에서 틀어진 각도 오차가 0.5도 미만이어야 한다.


<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/296a3b91-9e6a-4e85-86ec-c1ec71a3e834" />
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/b4655293-d00f-4a7c-ac20-5e203925b05b" />


## 4. Dataset

https://pan.baidu.com/s/1PoMSfgZQMnUGlhi7S5fFZw?pwd=2ik6

pretrained model

last.ckpt 구글 드라이브 다운로드


## 5. References

본 프로젝트는 아래 논문의 연구 성과를 바탕으로 구현 및 개선되었습니다.

[E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](https://ieeexplore.ieee.org/document/10588551)



