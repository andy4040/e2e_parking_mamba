🚗 Mamba 기반 End-to-End 자율 주차 시스템
CARLA 시뮬레이터 환경에서 Mamba(State Space Model) 아키텍처를 활용하여 구현한 효율적인 End-to-End 자율 주차 에이전트입니다.

🌟 프로젝트 핵심 요약
본 프로젝트는 기존의 Transformer 구조보다 연산 효율이 뛰어난 **Mamba(Selective State Space Model)**를 활용하여 주행 제어 명령을 직접 생성합니다. 복잡한 주차 시나리오에서 센서 데이터를 입력받아 조향(Steering), 가속(Acceleration), 제동(Brake) 값을 실시간으로 예측합니다.

Key Features
Mamba Backbone: 시퀀스 데이터 처리에 최적화된 Mamba 블록을 사용하여 장기 의존성(Long-range dependency) 문제를 해결하고 추론 속도를 향상했습니다.

End-to-End Learning: 원시 센서 데이터로부터 제어 값까지 단일 신경망으로 연결하여 복잡한 규칙 기반 시스템 없이 주차 로직을 학습했습니다.

<img width="776" height="324" alt="image" src="https://github.com/user-attachments/assets/ce9144c7-e88b-4b79-9458-55c2d8440b30" />



