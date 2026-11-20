# 2026-2 자연어처리

## Project 1. Seq2Seq Num2Text

숫자, 영어, 한글 사이의 "숫자 읽기"를 번역하는 GRU 기반 Seq2Seq 모델입니다.
6개 방향(num↔en, num↔ko, en↔ko)에 대해 각각 모델을 학습했습니다.

### 파일 구성
| 파일 | 설명 |
|---|---|
| `project01/num2text.ipynb` | 전체 코드 (데이터 처리, 모델, 학습, 평가, 테스트) |
| `project01/num2text_models.pt` | 학습된 6개 방향 모델 + 어휘 사전 |
| `project01/num2txt.txt` | 숫자 ↔ 영어 데이터 (10만 개) |
| `project01/num2kor.txt` | 숫자 ↔ 한글 데이터 (10만 개) |

### 데이터 처리
- 토큰화: 숫자와 한글은 글자 단위, 영어는 단어 단위
- 숫자는 정수(int)가 아닌 문자열로 처리
- 영어 ↔ 한글 데이터: 두 파일에 공통으로 있는 숫자 12,738개로 구성
- train / val / test = 8 : 1 : 1
- 오버샘플링: 학습 데이터에서 3자리 이하 숫자와 0이 4개 이상인 숫자를 10배로 늘림

### 모델
- Encoder: Embedding + 2-layer GRU
- Decoder: Embedding + 2-layer GRU + Linear
- Attention (선택): Luong 방식 (general score)
- 디코딩: Greedy decoding
- 학습: Adam (lr=1e-3), teacher forcing 0.5, gradient clipping 1.0
- val loss가 가장 낮은 epoch의 모델을 저장

### 모델 선택
각 방향마다 Attention 없는 모델(base)과 Attention 모델(attn)을 모두 학습하고,
**validation 정확도**가 더 높은 모델을 최종 모델로 선택했습니다.

### 결과 (문장 단위 완전 일치 정확도)
| 방향 | base val | base test | attn val | attn test | 선택 |
|---|---|---|---|---|---|
| num → en | 99.90% | 99.88% | 99.33% | 99.23% | base |
| en → num | 99.90% | 99.96% | 99.91% | 99.95% | attn |
| num → ko | 99.82% | 99.77% | 99.92% | 99.92% | attn |
| ko → num | 99.97% | 99.96% | 100.00% | 99.97% | attn |
| en → ko | 99.76% | 99.45% | 99.61% | 99.53% | base |
| ko → en | 99.61% | 99.45% | 99.69% | 99.61% | attn |

### 한계
- 1000001처럼 중간에 0이 많이 연속되는 숫자는 학습 데이터에 거의 없어서 틀리는 경우가 있습니다.

### 사용 방법
`num2text.ipynb`의 마지막 셀만 실행하면 저장된 모델을 불러와 테스트할 수 있습니다. (학습 셀 실행 불필요)

```python
num2en("12345")              # twelve thousand three hundred forty five
en2num("twelve thousand three hundred forty five")
num2ko("12345")              # 만 이천삼백사십오
ko2num("만 이천삼백사십오")
en2ko("twelve thousand three hundred forty five")
ko2en("만 이천삼백사십오")
```
