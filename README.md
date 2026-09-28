### 안녕하세요, 최민준(Minjoon Choi)입니다

전자전기공학 2학년입니다. **온디바이스 AI** — GPU 서버나 클라우드 없이, 사용자의 기기 안에서
모델을 빠르고 믿을 수 있게 돌리는 일 — 에 관심이 있습니다.

지금까지 만든 것은 노트북 한 대에서 오프라인으로 도는 인식 파이프라인 두 개입니다.
둘 다 제 교재와 강의를 AI에게 읽히려고 만든 도구이고, 설계 판단은 **실측 A/B로 정했습니다.**

---

#### [pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook) — 교재 PDF → 수식까지 살린 Markdown

- 수식 인식 모델의 ONNX 디코더에 KV캐시가 없고 PyTorch 가중치는 비공개여서, ONNX 가중치를
  `transformers` 모델에 이식해 **KV캐시 포함으로 다시 수출하고 int8로 양자화** → 인식 **4.16배**
- 인코더는 내장 GPU(DirectML), int8 디코더·수식 검출은 CPU — 장치 배치를 재 보고 정함
- 실제 교재 9권 7,530쪽, 원본 대조 **문자 일치율 93.8% · 낱말 회수율 96.5%**, 골든 테스트 290건

#### [lecture-transcriber](https://github.com/mijnch/lecture-transcriber) — 강의 영상 → 화면까지 읽은 타임스탬프 Markdown

- 노트북 CPU만으로(CUDA 없음) Whisper large-v3-turbo를 CTranslate2 int8로, 화면 OCR은 코어 수만큼 병렬로(**5.8배**, 출력 동일)
- 화면 전환을 '바뀐 화소 비율'로 찾아 합성 강의 슬라이드 **48/48장을 ±1초**로, 실강의 쪽 전환 **26/26**
- 말소리는 있는데 낱말이 없는 구간을 찾아 언어를 새로 정해 다시 읽는 등, **조용한 누락을 드러난 누락으로** 바꿈
- 단위 검증 152개 + 정답을 아는 합성 강의로 끝까지 돌려 채점하는 종단 검증 47항목

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
A/B comparisons.

- **[pdf-ocr-korean-textbook](https://github.com/mijnch/pdf-ocr-korean-textbook)** — Korean textbook
  PDFs to Markdown with LaTeX math. Rebuilt the formula recognizer with a KV cache and int8
  quantization by transplanting its ONNX weights into a `transformers` model (4.16× faster
  recognition); encoder on the iGPU via DirectML. 7,530 pages from 9 books, 93.8% character
  accuracy / 96.5% word recall against hand-transcribed pages.
- **[lecture-transcriber](https://github.com/mijnch/lecture-transcriber)** — recorded lectures to
  timestamped Markdown that includes what was on screen. CPU-only int8 Whisper; slide changes caught
  48/48 within ±1 s on a synthetic lecture; silent omissions turned into marked ones.

Contact: mijnch@gmail.com

</details>
