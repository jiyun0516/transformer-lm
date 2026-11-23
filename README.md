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

구현 예정 단계이다. Attention을 기반으로 RMSNorm, SwiGLU, Transformer 블록과 전체 언어 모델을 구성하고, cross-entropy 손실 및 토큰 생성 기능을 구현한다.

---

## Milestone 3 · Training

구현 예정 단계이다. 학습·검증 데이터 분리, 배치 구성, 검증 손실 측정, 체크포인트 저장을 학습 루프로 연결한다.

---

## Getting Started

Python 가상환경에서 의존성을 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch pytest numpy
```
