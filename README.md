# 📂 PatchCore Colab 산출물 파일 설명

## 1. `wrn50_l2l3.onnx`
- **무엇?**  
  WideResNet-50-2 모델을 **ONNX 형식**으로 변환한 파일
  이미지를 넣으면 **layer2, layer3 특징맵**을 뽑아줍니다.
- **왜 필요?**  
  C#에서는 PyTorch를 못 쓰니까, ONNX Runtime을 통해 이 모델을 실행해서  
  PatchCore가 사용하는 “특징 벡터(feature embedding)”를 얻습니다.  
- **비유**: 원재료(이미지)를 반죽(특징맵)으로 만들어주는 “믹서기”.

---

## 2. `nnscorer_search_index.faiss`
- **무엇?**  
  학습 단계에서 만들어진 **FAISS 검색 인덱스**
  1024차원 임베딩 공간에서 “이 벡터가 정상군과 얼마나 가까운지” 계산하는 DB.
- **왜 필요?**  
  추론 시 새 이미지의 특징을 여기서 검색해서  
  “정상/비정상” 정도를 거리(score)로 판단합니다.  
- **비유**: 정상 샘플들이 들어있는 “카탈로그/도감”. 새 샘플이 이 도감과 비슷하면 정상.

---

## 3. `csharp_config.json`
- **무엇?**  
  **전처리 설정** 파일.  
  - 이미지 크기 (resize 256, crop 224)  
  - 정규화 mean/std 값  
  - 기타 inference 시 필요한 옵션들
- **왜 필요?**  
  ONNX 모델에 이미지를 넣을 때 반드시 같은 전처리를 해야  
  학습 시와 같은 embedding을 얻을 수 있습니다.  
- **비유**: 요리 레시피에서 “재료 손질법”.

---

## 4. `params_meta.json`
- **무엇?**  
  원래 Python PatchCore 모델에서 쓰던 파라미터 요약본.  
  - 보통은 projection matrix, bias 등이 들어있는데  
  - 현재 버전에선 projection이 비어 있어서 **그냥 1024차원 그대로** 씁니다.
- **왜 필요?**  
  혹시 projection이 있을 경우, C#에서 그대로 재현해야 하기 때문.  
  (이번 모델은 projection 없음 → 단순화됨)
- **비유**: “설계도”인데 이번에는 추가 공정이 없어서 빈칸.

---

## 5. `manifest.json`
- **무엇?**  
  내보내기 툴에서 만든 “메타 정보” 파일.  
  어떤 파일이 있고 어떤 버전인지 정리해 둔 관리용 문서.
- **왜 필요?**  
  꼭 없어도 동작은 하지만, 버전 추적/자동화 스크립트에서 유용합니다.  
- **비유**: “이 박스 안에는 어떤 부품이 들어있다”라는 포장 리스트.

---

## 6. `patchcore_params.pkl`
- **무엇?**  
  Python에서 쓰는 원래 PatchCore 모델 파라미터 (피클 형식).  
- **왜 필요?**  
  Colab/Python용 백업일 뿐, C#에서 직접 쓰진 않습니다.  
  (우린 ONNX + JSON + FAISS만 쓰면 됨)
- **비유**: 원래 원본 레시피 노트북. 직접 요리할 땐 안 보고, 참고용으로만 보관.

---

# ✅ 정리
- **C# 필수**:  
  `wrn50_l2l3.onnx`, `nnscorer_search_index.faiss`, `csharp_config.json`  
- **옵션/참고**:  
  `params_meta.json`, `manifest.json`  
- **Python 전용**:  
  `patchcore_params.pkl`  

즉, **실제 C# 추론 구현 시 반드시 필요한 파일은 3개**이고,  
나머지는 참고 자료/백업이에요.
