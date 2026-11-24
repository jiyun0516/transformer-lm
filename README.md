# Transformer LM - PyTorch 기반 언어 모델

> Attention 구현에서 시작해 Transformer 언어 모델과 학습 과정으로 확장한다.

## Project Overview

앞선 토큰을 바탕으로 다음 토큰을 예측하는 decoder-only Transformer를 PyTorch로 구현하는 프로젝트이다. Attention, 언어 모델, 학습 파이프라인의 세 단계로 개발한다.

---

## Milestone 1 · Attention

`starter/attention.py`에 causal multi-head self-attention을 구현했다. Softmax와 attention 연산을 직접 작성하고, causal mask와 RoPE를 결합했다.

| Component | Implementation |
| --- | --- |
| Stable Softmax | 지정한 축의 최댓값을 빼 지수 계산의 오버플로 방지 |
| Scaled Dot-Product Attention | Query와 key의 내적을 스케일링하고 value의 가중합 계산 |
| Causal Mask | 각 query가 현재와 이전 위치의 key만 참조하도록 제한 |
| Rotary Position Embedding | Query와 key의 인접 채널 쌍을 위치에 따라 회전시켜 상대적 위치 정보 반영 |
| Multi-Head Self-Attention | Q·K·V를 여러 head로 나누고, head별 attention 결과를 출력 projection으로 결합 |

공개 테스트 9개를 모두 통과했다. 테스트는 softmax의 수치적 안정성, attention 계산 결과, RoPE의 노름 보존과 상대 위치 성질, 출력의 인과성 등을 확인한다.

```bash
python -m pytest tests/test_m1_attention.py -q
```

---

## Milestone 2 · Language Model

`starter/transformer.py`에 decoder-only Transformer 언어 모델을 구현했다. 앞서 구현한 attention을 pre-norm 블록으로 구성하고, 토큰 임베딩부터 어휘별 점수 출력까지 연결했다. 입력 임베딩과 출력 projection은 별도의 가중치를 사용한다.

| Component | Implementation |
| --- | --- |
| RMSNorm | 마지막 차원의 제곱 평균을 이용해 fp32로 정규화하고 학습 가능한 가중치 적용 |
| SwiGLU | SiLU를 적용한 gate와 별도 선형 변환 결과를 원소별로 곱한 뒤 출력 차원으로 변환 |
| Transformer Block | Attention과 FFN 각각에 pre-norm과 residual 연결 적용 |
| Transformer LM | 토큰 임베딩, Transformer 블록, 최종 RMSNorm과 bias 없는 어휘 projection으로 구성 |
| Cross-Entropy | 최댓값을 뺀 log-sum-exp 방식으로 정답 토큰의 평균 손실 계산 |
| Autoregressive Generation | 마지막 위치의 logits로 다음 토큰을 선택하고 입력에 추가하는 과정을 반복 |

생성 시 temperature에 따른 샘플링과 seed 기반 재현성을 지원한다. 문맥이 설정된 길이를 넘으면 최근 `max_len`개 토큰을 사용하며, temperature가 0이면 가장 높은 점수의 토큰을 선택한다.

공개 테스트 11개를 모두 통과했다. 테스트는 정규화와 SwiGLU 계산, residual 연결, 모델 출력의 형태와 인과성, 손실의 수치적 안정성, 생성 결과의 재현성을 확인한다. 반복 시퀀스를 학습해 손실을 낮추는 테스트도 통과했다.

```bash
python -m pytest tests/test_m2_transformer.py -q
```

---

## Milestone 3 · Training

구현 예정 단계이다. 학습·검증 데이터 분리, 배치 구성, 검증 손실 측정, 체크포인트 저장을 학습 루프로 연결한다.

---

## Getting Started

Python 가상환경에서 의존성을 설치한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest numpy
```