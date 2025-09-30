# 📂 PatchCore Colab 산출물 파일 설명 (최종)

## 1. `wrn50_l3.onnx`
- **무엇?**  
  WideResNet-50-2 백본을 ONNX로 내보낸 파일.  
  이미지를 넣으면 **layer3 특징맵**을 출력합니다 (출력: `[1,1024,14,14]`).
- **왜 필요?**  
  PyTorch를 쓸 수 없는 .NET 환경에서 ONNX Runtime을 통해 임베딩을 얻어야 합니다.  
  PatchCore는 이 layer3 임베딩을 기반으로 anomaly score를 계산합니다.
- **비유**: 원재료(이미지)를 “밀가루 반죽(특징맵)”으로 바꿔주는 믹서기.

---

## 2. `gallery_f32.bin`
- **무엇?**  
  학습 데이터에서 추출한 정상 샘플들의 **feature 메모리**.  
  float32 배열로, `(N×1024)` 크기의 row-major 구조이며 각 행은 L2 정규화되어 있음.
- **왜 필요?**  
  새 이미지가 들어오면, 이 갤러리와 코사인 거리(1 − dot)를 비교하여  
  “정상과 얼마나 가까운지”를 판단합니다.
- **비유**: 정상 샘플들이 차곡차곡 들어있는 “참고 도감”.

---

## 3. `threshold.json`
- **무엇?**  
  학습 데이터(Good set)의 분포를 기반으로 산출된 **임계값**.  
  - `"value"`: 스코어 기준치 (예: 0.0509)  
  - `"metric"`: 거리 방식 (`"cosine"`)
- **왜 필요?**  
  추론 시 `score > threshold` → **NotGood**,  
  `score ≤ threshold` → **Good** 으로 판정합니다.
- **비유**: 시험 합격선 같은 “커트라인”.

---

## 4. `meta.json`
- **무엇?**  
  모델 실행/전처리 환경 메타데이터.  
  - `input_size` (224)  
  - `mean/std` (ImageNet 값)  
  - `backbone` (`wrn50_l3`)  
- **왜 필요?**  
  Colab과 C#에서 동일한 전처리를 맞추기 위한 기준서.
- **비유**: 요리 레시피의 “재료 손질법”.

---

# ✅ 정리
- **C# 필수** (반드시 있어야 함):  
  - `wrn50_l3.onnx`  
  - `gallery_f32.bin`  
  - `threshold.json`  
  - `meta.json`  

- **Python/백업 전용 (C#에서 안 씀)**:  
  - `patchcore_params.pkl` (원래 PyTorch 모델 파라미터)  
  - `nnscorer_search_index.faiss` (Python용 검색 인덱스)  
  - 기타 manifest, exporter 로그  

즉, **C# 추론·오버레이 구현에 필요한 건 딱 4개**입니다.  
