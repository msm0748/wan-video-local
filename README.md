# Wan2.2 Mac M4 설정 가이드

이 프로젝트는 Mac M4 (통합 메모리 24GB) 환경에서 `uv`와 `diffusers`를 사용하여 Wan2.2 (5B) 비디오 생성 모델을 구동하기 위한 설정을 담고 있습니다.

## 필수 사양

- **하드웨어**: Mac M4 (RAM 24GB 이상 권장)
- **저장 공간**: 모델 가중치를 위해 최소 15GB 이상의 여유 공간
- **도구**: `uv` (Homebrew 또는 pip로 설치)

## 설치 방법

1. **환경 초기화 및 의존성 설치**:

   ```bash
   uv sync
   ```

   > `pyproject.toml`에 명시된 `diffusers`, `torch`, `transformers` 등의 호환 버전을 자동으로 설치합니다.

## 실행 방법

텍스트 프롬프트를 기반으로 비디오를 생성하려면 다음 명령어를 실행하세요:

```bash
uv run python run_wan_5b.py
```

### 설정 변경

`run_wan_5b.py` 파일을 열어 다음 항목을 수정할 수 있습니다:

- `PROMPT`: 생성할 비디오의 텍스트 설명
- `OUTPUT_PATH`: 결과물 저장 경로
- **해상도 및 프레임**:
  - `height`: 480 (24GB 메모리 안전값)
  - `width`: 480 (24GB 메모리 안전값)
  - `num_frames`: 29 (프레임 수가 높으면 메모리 부족 발생 가능)

## 문제 해결 (Troubleshooting)

### 메모리 부족 (OOM) 발생 시

1. 실행 중인 다른 무거운 애플리케이션을 종료하세요.
2. 스크립트 내 `height`, `width` 해상도를 낮추세요.
3. `num_frames`를 줄이세요.
4. `run_wan_5b.py` 내의 메모리 최적화 옵션이 켜져 있는지 확인하세요:
   - `pipe.enable_attention_slicing()` (기본 활성화)

### `AttributeError: 'WanPipeline' object has no attribute 'enable_vae_slicing'` 오류

Wan2.2용 Diffusers 최신 버전에서는 `enable_vae_slicing`을 지원하지 않을 수 있습니다.
대신 `pipe.enable_attention_slicing()`을 사용하도록 코드가 이미 수정되어 있습니다.

## 모델 정보

- **모델 ID**: Wan-AI/Wan2.2-TI2V-5B-Diffusers
- **데이터 타입**: `bfloat16` (Apple Silicon 성능 및 메모리 최적화)
