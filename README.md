## 1. 프로젝트 요약
🚗 Mamba 기반 End-to-End 자율 주차 시스템 

본 프로젝트는 CARLA 시뮬레이터 환경에서 Transformer의 정밀한 특징 추출과 Mamba(State Space Model)의 효율적인 시퀀스 모델링을 결합한 하이브리드 End-to-End 주차 에이전트입니. 기존 아키텍처에 Mamba 블록을 통합하여, 복잡한 주차 시나리오에서 발생하는 고차원 센서 데이터를 제어 신호로 더욱 정밀하게 정제(Refinement)합니다.

🚀 Key Innovation: Mamba-Driven Context Refinement

기존 모델이 정적인 특징 추출에 그쳤다면, 본 프로젝트는 258차원의 고차원 특징 시퀀스를 Mamba 블록으로 재가공하여 주행 맥락을 극대화했습니다.
258-Feature Spatial Scanning: 융합된 BEV 특징으로부터 추출된 **258개의 핵심 정보 채널($d_{model}=258$)**을 시퀀스 데이터로 간주하고 정밀 스캔합니다. 이는 주차 칸의 미세한 각도 변화와 장애물과의 거리 관계를 시계열적으로 재해석하여 제어의 연속성을 보장합니다.

Selective Information Filtering (S6): 단순한 특징 전달이 아닌, S6 메커니즘을 통해 258개의 정보 중 현재 제어(Steer, Accel)에 가장 유효한 정보만을 선택적으로 기억하고 노이즈를 억제합니다.

High-Speed Parallel Processing: 병렬 스캔(Parallel Scan) 기법을 통해 시퀀스 길이에 대해 **$O(\log L)$**의 복잡도를 실현, 실시간 제어 환경에서 지연 시간(Latency)을 최소화했습니다.



##2. 시스템 아키텍쳐

1. Multi-View Input (4 Surround Images): 전/후/좌/우 4개 카메라 데이터 입력

2. BEV Feature Extraction:2D 이미지를 Bird's-Eye View 특징 맵으로 변환

3. Feature Fusion: 차량 상태 및 목표 지점 결합 (Vehicle & Target Point Integration)

4. ✨ Mamba Sequence Modeling (Added): 융합된 258개의 특징 벡터를 입력으로 받아 전역적 맥락을 스캔.Simplified S6 구조 적용: $B$와 $C$ 가중치를 공유하여 파라미터 효율성을 높이고 학습 안정성 확보.단순한 특징 결합을 넘어, 시퀀스 모델링을 통해 '공간적 배치'를 '주행 의도'로 정제하는 핵심 단계.

5. Control Prediction: 조향(Steering), 가속(Accel), 제동(Brake) 신호 생성

<img width="1184" height="660" alt="image" src="https://github.com/user-attachments/assets/450ce9db-2abe-47c0-ac0a-155d77e19d51" />




## 3. Setup
```
git clone https://github.com/andy4040/e2e_parking_mamba.git
cd e2e-parking-carla/
conda env create -f environment.yml
conda activate E2EParking
chmod +x setup_carla.sh
./setup_carla.sh
```
```
./carla/CarlaUE4.sh -opengl -ResX=160 -ResY=120 -fps=10
```
Evaluate
```
python3 carla_parking_eva.py
```
Training

```
python pl_train.py 
```
## 4. Dataset & Pretrained Model

Dataset

https//pan.baidu.com/s/1PoMSfgZQMnUGlhi7S5fFZw?pwd=2ik6

pretrained model

https://drive.google.com/file/d/1gXtrB8Ub9_6D7LWa1YCOB0HKcJlOZ3Hz/view?usp=drive_link

## 5. 결과

NVIDIA RTX 2080 GPU 환경에서 총 30 Epoch 동안 약 60시간의 학습을 진행하였다.

🏁 주차 성공 판정 기준 


차량의 중심이 주차 칸의 정중앙으로부터 0.5m 이내에 위치해야 하며, 차체가 목표 방향에서 틀어진 각도 오차가 0.5도 미만이어야 한다.


<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/296a3b91-9e6a-4e85-86ec-c1ec71a3e834" />
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/b4655293-d00f-4a7c-ac20-5e203925b05b" />





## 6. References

본 프로젝트는 아래 논문의 연구 성과를 바탕으로 구현 및 개선되었습니다.

[E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](https://ieeexplore.ieee.org/document/10588551)



