import os
import os 

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from .latentsync.models.unet import UNet3DConditionModel
from .latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from .latentsync.whisper.audio2feature import Audio2Feature


import torch
import os 

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

class CustomLatentSync:
    def __init__(self, unet_config_path, inference_ckpt_path):
        self.unet_config_path = unet_config_path
        self.inference_ckpt_path = inference_ckpt_path
        self.mask_image_path = "LatentSync/latentsync/utils/mask.png"

        self.config = OmegaConf.load(self.unet_config_path)

        # Dynamically choose the best available device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32  # MPS does not support float16 well
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        scheduler = DDIMScheduler.from_pretrained("LatentSync/configs")

        whisper_model_path = "LatentSync/weights/tiny.pt"

        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device=self.device,
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )

        print("Loaded Audio Encoder")

        print("Using device: ", self.device)

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=self.dtype
        ).to(self.device)
        
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        print("Loaded VAE")

        denoising_unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            self.inference_ckpt_path,
            device=self.device,
        )

        denoising_unet = denoising_unet.to(dtype=self.dtype)

        print("Loaded UNET Model")

        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        ).to(self.device)

        if self.config.run.seed != -1:
            set_seed(self.config.run.seed)
        else:
            torch.seed()

        print(f"Initial seed: {torch.initial_seed()}")

    def infer(self, video_path, audio_path, video_out_path, inference_steps, guidance_scale):
        video_mask_path = video_out_path.replace(".mp4", "_mask.mp4")
        self.pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            video_mask_path=video_mask_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=self.dtype,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask_image_path=self.mask_image_path,
        )

