### 안녕하세요, 최민준(Minjoon Choi)입니다

전자전기공학 2학년입니다. **온디바이스 AI** — GPU 서버나 클라우드 없이, 사용자의 기기 안에서
모델을 빠르고 믿을 수 있게 돌리는 일 — 에 관심이 있습니다.

지금까지 만든 것은 노트북 한 대에서 오프라인으로 도는 인식 파이프라인 두 개입니다.
둘 다 제 교재와 강의를 AI에게 읽히려고 만든 도구이고, 설계 판단은 **실측 A/B로 정했습니다.**
코드는 AI 코딩 도구(Claude Code)와 함께 작성했고, 문제 정의·실측·채택 판단은 제가 했습니다.

세 번째 저장소는 int8 양자화가 실제로 어떤 정수 연산인지 확인하는 실험입니다.
계획은 제가 세웠고, 구현·측정·문서는 Claude Code가 수행했습니다.

---

#### [pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook) — 교재 PDF → 수식까지 살린 Markdown

- 수식 인식 모델의 ONNX 디코더에 KV캐시가 없고 PyTorch 가중치는 비공개여서, ONNX 가중치를
  `transformers` 모델에 이식해 **KV캐시 포함으로 다시 수출하고 int8로 양자화** → 인식 **4.16배**
- 인코더는 내장 GPU(DirectML)로 옮겨 **전체 1.37배**, int8 디코더·수식 검출은 CPU — 장치 배치를 재 보고 정함
- 실제 교재 9권 7,530쪽, 원본 대조 **문자 일치율 93.8% · 낱말 회수율 96.5%**, 골든 테스트 290건

#### [lecture-transcriber](https://github.com/mijnch/lecture-transcriber) — 강의 영상 → 화면까지 읽은 타임스탬프 Markdown

- 노트북 CPU만으로(CUDA 없음) Whisper large-v3-turbo를 CTranslate2 int8로, 화면 OCR은 코어 수만큼 병렬로(**5.8배**, 출력 동일)
- 화면 전환을 '바뀐 화소 비율'로 찾아 합성 강의 슬라이드 **48/48장을 ±1초**로, 실강의 쪽 전환 **26/26**
- 말소리는 있는데 낱말이 없는 구간을 찾아 언어를 새로 정해 다시 읽는 등, **조용한 누락을 드러난 누락으로** 바꿈
- 단위 검증 152개 + 정답을 아는 합성 강의로 끝까지 돌려 채점하는 종단 검증 47항목

#### [int8-quant-lab](https://github.com/mijnch/int8-quant-lab) — INT8 양자화를 라이브러리 출력과 비트 단위로 맞춰 보기

- MLPerf Tiny 키워드 인식 모델(DS-CNN)에서 TF 2.21 변환기가 정한 scale·zero-point와 int8 값(가중치 22,016개·bias 588개)을 **전부 똑같이 재현**
- 정수 추론을 numpy로 다시 구현해 LiteRT 세 실행 경로(참조·최적화·XNNPACK)와 테스트 4,890개 전체에서 **불일치 0** — 같은 모델도 경로마다 반올림 규칙이 달라, 5가지를 원문에서 찾아 맞춤
- FP32 92.17% → INT8 92.33%(통계적으로 구별되지 않음), 파일 크기 절반 · 4비트 텐서별 대칭 가중치는 57.4%로 붕괴(시뮬레이션)
- 계획(범위·지표·정직성 원칙)은 제가 세웠고, **구현·측정·문서는 Claude Code가 수행**

---

**다루어 본 것** · Python · ONNX Runtime (DirectML) · CTranslate2 / faster-whisper · PyTorch · transformers ·
optimum (ONNX 수출) · 동적 int8 양자화 · Tesseract · ffmpeg · pdfium · NumPy · GitHub Actions

**연락** · mijnch@gmail.com

<details>
<summary>In English</summary>

I'm Minjoon Choi, a second-year Electrical and Electronic Engineering student interested in
**on-device AI** — running models fast and reliably on the user's own device, without a GPU server
or the cloud.

So far I've built two offline recognition pipelines that run on a single laptop, both made so that an
AI assistant could read my own textbooks and lectures. Design decisions in them were settled by measured
A/B comparisons. The code was written with an AI coding assistant (Claude Code); problem definition,
measurement, and the decision to adopt each design are mine.

- **[pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook)** — Korean textbook
  PDFs to Markdown with LaTeX math. Rebuilt the formula recognizer with a KV cache and int8
  quantization by transplanting its ONNX weights into a `transformers` model (4.16× faster
  recognition); encoder on the iGPU via DirectML (1.37× end-to-end). 7,530 pages from 9 books, 93.8% character
  accuracy / 96.5% word recall against hand-transcribed pages.
- **[lecture-transcriber](https://github.com/mijnch/lecture-transcriber)** — recorded lectures to
  timestamped Markdown that includes what was on screen. CPU-only int8 Whisper; slide changes caught
  48/48 within ±1 s on a synthetic lecture; silent omissions turned into marked ones.
- **[int8-quant-lab](https://github.com/mijnch/int8-quant-lab)** — TFLite/LiteRT INT8 quantization
  re-implemented in NumPy and matched bit-for-bit: every scale, zero-point and int8 value the TF 2.21
  converter produced for the MLPerf Tiny KWS model, and the integer outputs of three execution paths
  (reference, optimized, XNNPACK) on the full test set, which round in five different ways. The plan is
  mine; implementation, measurement and documentation were done by Claude Code.

Contact: mijnch@gmail.com

</details>
