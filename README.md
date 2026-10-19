### 안녕하세요, 최민준(Minjoon Choi)입니다

홍익대학교 전자전기공학부 2학년입니다. **온디바이스 AI** — GPU 서버나 클라우드 없이, 사용자의 기기 안에서
모델을 빠르고 믿을 수 있게 돌리는 일 — 에 관심이 있습니다.
관심 연구는 **저비트 양자화·모델 압축**(온디바이스 추론)입니다.

양자화 실험 두 개(int8-quant-lab, lowbit-qer-lab)는 계획과 정직성 원칙을 제가 세웠고,
구현·측정·문서는 Claude Code가 수행했습니다.

그 밖에 만든 것은 노트북 한 대에서 오프라인으로 도는 도구 세 개입니다.
모두 제 교재·강의·시험지를 AI에게 읽히려고 만든 것이고, 설계 판단은 **실측 A/B로 정했습니다.**
코드는 AI 코딩 도구(Claude Code)와 함께 작성했고, 문제 정의·실측·채택 판단은 제가 했습니다.

---

#### [int8-quant-lab](https://github.com/mijnch/int8-quant-lab) — INT8 양자화를 라이브러리 출력과 비트 단위로 맞춰 보기

- MLPerf Tiny 키워드 인식 모델(DS-CNN)에서 TF 2.21 변환기가 정한 scale·zero-point와 int8 값(가중치 22,016개·bias 588개)을 **전부 똑같이 재현**
- 정수 추론을 numpy로 다시 구현해 LiteRT 세 실행 경로(참조·최적화·XNNPACK)와 테스트 4,890개 전체에서 **불일치 0** — 같은 모델도 실행 경로·플랫폼마다 반올림 규칙이 달라, 6가지를 원문에서 찾아 맞춤
- FP32 92.17% → INT8 92.33%(통계적으로 구별되지 않음), 파일 크기 절반 · 4비트 텐서별 대칭 가중치는 57.4%로 붕괴(시뮬레이션)
- [왜 무너졌나](https://github.com/mijnch/int8-quant-lab/blob/main/docs/analysis_4bit.md): 오차 크기는 Δ²/12 모델 그대로(0.2 dB 안). 원인은 큰 가중치가 거친 격자에 오르는 것 + 특정 층의 오차가 **뒤쪽 망을 지나며 증폭**되는 것(logits SQNR 9.0 dB, 다른 층 13.7~28.9) — 가중치 MSE는 민감한 층을 고르지 못함
- 도구: NumPy · LiteRT(ai-edge-litert) · TensorFlow 2.21(변환) · flatbuffers · McNemar 검정 · GitHub Actions(Linux/Windows)

#### [lowbit-qer-lab](https://github.com/mijnch/lowbit-qer-lab) — 저비트 양자화 오차의 저랭크 복원(W ≈ Q + LR), numpy 단순화 구현

- RTN(그룹 128) · QER · 보존-복원 랭크 분할 · 대리지표를 구현해 **SmolLM2-135M·TinyLlama-1.1B의 실제 선형층 49개**(3·4비트)에서 대조: 분할이 QER보다 가중치 오차를 줄이는 정도는 **ρ_r(W) 하나로 결정**(상관 −0.97/−0.98 — 첫 층 q/k에서 41~61%↓, TinyLlama 층 0 q_proj 0.247 → 0.101), 최적은 거의 항상 k=r(전부 보존, 복원 0), 대리지표 후회 ≤3.9%
- **perplexity(WikiText-2)는 가중치 오차와 반대로 움직임**: 분할이 가중치 오차는 더 낮은데 QER이 훨씬 낫고(3비트 80.9 대 121.1), 층 하나씩 실제 활성값으로 잰 출력 오차로도 설명되지 않음(열린 질문) · 같은 비트 예산에서는 한 비트 더 쓰는 RTN이 둘 다 이김(4비트 RTN 22.7)
- 논문 재현 아님(양자화기·스케일·탐침을 단순화) · 테스트 58개(Eckart–Young, k=0↔QER, 오차 한계) · CI Ubuntu·Windows
- 도구: NumPy · SVD/저랭크 근사 · safetensors 직접 읽기(torch 불필요) · PyTorch/transformers(perplexity) · GitHub Actions

#### [pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook) — 교재 PDF → 수식까지 살린 Markdown

- 수식 인식 모델의 ONNX 디코더에 KV캐시가 없고 PyTorch 가중치는 비공개여서, ONNX 가중치를
  `transformers` 모델에 이식해 **KV캐시 포함으로 다시 수출하고 int8로 양자화** → 인식 **4.16배**
- 인코더는 내장 GPU(DirectML)로 옮겨 **전체 1.37배**, int8 디코더·수식 검출은 CPU — 장치 배치를 재 보고 정함
- 실제 교재 9권 7,530쪽, 원본 대조 **문자 일치율 93.8% · 낱말 회수율 96.5%**, 골든 테스트 290건
- 도구: ONNX Runtime(DirectML) · PyTorch · transformers · optimum(ONNX 수출) · 동적 int8 양자화 · Tesseract · pdfium · GitHub Actions(Windows)

#### [lecture-transcriber](https://github.com/mijnch/lecture-transcriber) — 강의 영상 → 화면까지 읽은 타임스탬프 Markdown

- 노트북 CPU만으로(CUDA 없음) Whisper large-v3-turbo를 CTranslate2 int8로, 화면 OCR은 코어 수만큼 병렬로(**5.8배**, 출력 동일)
- 화면 전환을 '바뀐 화소 비율'로 찾아 합성 강의 슬라이드 **48/48장을 ±1초**로, 실강의 쪽 전환 **26/26**
- 말소리는 있는데 낱말이 없는 구간을 찾아 언어를 새로 정해 다시 읽는 등, **조용한 누락을 드러난 누락으로** 바꿈
- 단위 검증 152개 + 정답을 아는 합성 강의로 끝까지 돌려 채점하는 종단 검증 47항목
- 도구: faster-whisper(CTranslate2 int8) · ffmpeg · Tesseract(병렬) · GitHub Actions

#### [hwp-to-pdf](https://github.com/mijnch/hwp-to-pdf) — 한글 문서(.hwp/.hwpx) → 한컴 없이, 글자까지 맞는 PDF

- 오픈소스 렌더러 rhwp의 PDF는 보기엔 맞지만 글자층이 깨져(pdfium 한글 60~100%, pdfminer 띄어쓰기 0%), 쪽 Form을 펼치고 보이지 않는 공백을 넣어 **추출기 4종 모두 한글 100%** — 그림은 그대로
- 한컴 원본(문서에 든 1쪽 미리보기, 한컴이 내보낸 PDF 34쌍 615쪽)과 맞대 rhwp 조판을 **8곳 고쳐** 실제 시험지 1쪽 실루엣 일치율 **79.9 → 98.6%**, 8건 모두 상류에 이슈로 올림
- 변환마다 한컴 미리보기와 스스로 대조해 다르면 "확인 권장" · 157쪽 24초(쪽마다 동시에 그려도 한 번에 내보낸 것과 화소 동일)
- 단위 검증 90개 + 합성 문서 종단 검증 30항목 + rhwp 공개 시험 문서 312개
- 도구: rhwp(Rust) 소스 패치·빌드 · pikepdf · pypdfium2 · GitHub Actions(Windows)

---

**연락** · mijnch@gmail.com

<details>
<summary>In English</summary>

I'm Minjoon Choi, a second-year Electrical and Electronic Engineering student at Hongik University,
interested in **on-device AI** — running models fast and reliably on the user's own device, without a
GPU server or the cloud. Research interest: **low-bit quantization and model compression** for
on-device inference.

Two quantization experiments (int8-quant-lab, lowbit-qer-lab): the plan and the honesty rules are
mine; implementation, measurement and documentation were done by Claude Code.

Besides those, I've built three offline tools that run on a single laptop, all made so that an AI
assistant could read my own textbooks, lectures and exam papers. Design decisions in them were settled
by measured A/B comparisons. The code was written with an AI coding assistant (Claude Code); problem
definition, measurement, and the decision to adopt each design are mine.

- **[int8-quant-lab](https://github.com/mijnch/int8-quant-lab)** — TFLite/LiteRT INT8 quantization
  re-implemented in NumPy and matched bit-for-bit: every scale, zero-point and int8 value the TF 2.21
  converter produced for the MLPerf Tiny KWS model, and the integer outputs of three execution paths
  (reference, optimized, XNNPACK) on the full test set, which round in six different ways across paths
  and platforms. FP32 92.17% → INT8 92.33% (statistically indistinguishable); 4-bit per-tensor
  symmetric weights collapse to 57.4% (simulated) — and the [analysis](https://github.com/mijnch/int8-quant-lab/blob/main/docs/analysis_4bit.md)
  shows the weight error is exactly what the Δ²/12 model predicts; the damage comes from large weights
  on a coarse grid and from one layer's error being amplified on the way to the logits.
  Tools: NumPy, LiteRT, TensorFlow 2.21 (conversion), flatbuffers, GitHub Actions (Linux/Windows).
- **[lowbit-qer-lab](https://github.com/mijnch/lowbit-qer-lab)** — low-bit weight quantization with
  low-rank error reconstruction (W ≈ Q + LR) in NumPy: group-128 RTN, QER, a preserve-then-quantize
  rank split and a surrogate for choosing the split, compared on 49 real linear layers of
  SmolLM2-135M and TinyLlama-1.1B. The split's gain over QER in weight error is set by a single number,
  ρ_r(W) (correlation −0.97/−0.98; first-layer q/k projections gain 41–61%, TinyLlama layer-0 q_proj
  0.247 → 0.101), the optimum is almost always k = r, and surrogate regret is ≤ 3.9%. Perplexity on
  WikiText-2 moves the other way: the split has the lower weight error yet QER is far better (3-bit
  80.9 vs 121.1), and per-layer output error with real activations does not explain it — an open
  question; at equal bits per weight, one more bit of RTN beats both (4-bit RTN 22.7). Not a paper
  reproduction. 58 tests. Tools: NumPy, SVD, a torch-free safetensors reader, PyTorch/transformers.
- **[pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook)** — Korean textbook
  PDFs to Markdown with LaTeX math. Rebuilt the formula recognizer with a KV cache and int8
  quantization by transplanting its ONNX weights into a `transformers` model (4.16× faster
  recognition); encoder on the iGPU via DirectML (1.37× end-to-end). 7,530 pages from 9 books, 93.8% character
  accuracy / 96.5% word recall against hand-transcribed pages. Tools: ONNX Runtime (DirectML), PyTorch,
  transformers, optimum, Tesseract, pdfium.
- **[lecture-transcriber](https://github.com/mijnch/lecture-transcriber)** — recorded lectures to
  timestamped Markdown that includes what was on screen. CPU-only int8 Whisper; slide changes caught
  48/48 within ±1 s on a synthetic lecture; silent omissions turned into marked ones. Tools:
  faster-whisper (CTranslate2 int8), ffmpeg, Tesseract.
- **[hwp-to-pdf](https://github.com/mijnch/hwp-to-pdf)** — Hangul word-processor files (.hwp/.hwpx)
  to PDF without Hancom Office. Repaired the open-source renderer rhwp's broken text layer (100% Hangul
  across four extractors) and, by comparing against Hancom's own renderings, fixed eight layout bugs in
  rhwp (page-1 silhouette agreement on real exam papers 79.9% → 98.6%; all reported upstream). Each
  conversion is self-checked against the preview image Hancom stores in the file. Tools: rhwp (Rust,
  patched and built from source), pikepdf, pypdfium2.

Contact: mijnch@gmail.com

</details>
