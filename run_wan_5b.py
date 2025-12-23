import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# 설정
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = "Thinking man, 3d render, high quality, 4k"
OUTPUT_PATH = "output.mp4"

def generate_video():
    print(f"모델 로딩 중: {MODEL_ID}...")
    
    # 1. 파이프라인 로드 (bfloat16 사용)
    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    # 2. Mac M4 (MPS) 최적화 설정
    device = torch.device("mps")
    pipe.to(device)

    # 3. 메모리 절약 기술 활성화 (24GB 환경 필수)
    print("메모리 최적화 활성화 중...")
    # VAE 단계에서 메모리 부족을 방지하기 위해 타일링과 슬라이싱을 켭니다.
    # WanPipeline에서는 아직 enable_vae_slicing, enable_vae_tiling을 지원하지 않을 수 있습니다.
    # 대신 enable_attention_slicing을 사용합니다.
    # pipe.enable_vae_slicing()
    # pipe.enable_vae_tiling()
    
    # 주의: Mac에서 enable_model_cpu_offload()는 때때로 mps와 충돌할 수 있습니다.
    # 대신 필요하다면 아래와 같이 attention 최적화를 시도할 수 있습니다.
    pipe.enable_attention_slicing()

    print("비디오 생성 시작 (이 작업은 시간이 다소 소요될 수 있습니다)...")
    try:
        video = pipe(
            prompt=PROMPT,
            height=480,        # 24GB에서는 이 해상도가 안전합니다.
            width=480,
            num_frames=29,      # 프레임 수가 많을수록 메모리 점유가 급증합니다.
            num_inference_steps=30, # 테스트를 위해 단계를 30으로 낮추는 것을 추천합니다.
            generator=torch.Generator(device="mps").manual_seed(42)
        ).frames[0]

        print(f"비디오 저장 중: {OUTPUT_PATH}...")
        export_to_video(video, OUTPUT_PATH, fps=16)
        print("작업 완료!")
        
    except RuntimeError as e:
        print(f"메모리 부족 또는 오류 발생: {e}")
        print("팁: num_frames를 줄이거나 해상도를 더 낮추어 보세요.")

if __name__ == "__main__":
    generate_video()