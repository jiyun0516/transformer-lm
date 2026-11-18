# Transformer LM - PyTorch 기반 언어 모델

> Causal attention부터 Transformer 블록과 학습 루프까지 단계별로 구현합니다.

## Project Overview

앞선 토큰을 바탕으로 다음 토큰을 예측하는 decoder-only Transformer를 PyTorch로 구현하는 프로젝트입니다.

현재는 causal multi-head self-attention을 구현했습니다. 각 토큰은 자신과 이전 위치의 토큰을 참고하며, RoPE를 사용해 상대적인 위치 차이를 attention 점수에 반영합니다.

---

## Implementation Scope

| Component | Implementation |
| --- | --- |
| Stable Softmax | 지정한 축의 최댓값을 빼 지수 계산의 오버플로 방지 |
| Scaled Dot-Product Attention | Query와 key의 내적을 스케일링하고 value의 가중합 계산 |
| Causal Mask | 각 query가 현재와 이전 위치의 key만 참조하도록 제한 |
| Rotary Position Embedding | Query와 key의 인접 채널 쌍을 토큰 위치에 따라 회전 |
| Multi-Head Self-Attention | 여러 head에서 attention을 계산하고 출력 projection으로 결합 |

---

## Development Status

| Stage | Scope | Status |
| --- | --- | --- |
| Milestone 1 | Causal attention과 RoPE | 완료 · 공개 테스트 9개 통과 |
| Milestone 2 | Transformer 블록과 토큰 생성 | 예정 |
| Milestone 3 | 학습 루프, 검증 손실, 체크포인트 | 예정 |

---

## Testing

저장소 루트에서 실행합니다.

```bash
python -m pip install torch pytest numpy
python -m pytest tests/test_m1_attention.py -q

### 2. 커밋 만들기

**Git 초기화와 `.gitignore` 작성을 아직 안 했다면** 아래를 실행해.

```bash
git init -b main

cat >> .gitignore <<'EOF'
.venv/
__pycache__/
.pytest_cache/
.DS_Store
README.assignment.md
