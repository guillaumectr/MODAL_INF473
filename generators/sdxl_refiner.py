import torch
from diffusers import DiffusionPipeline

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLRefinerGenerator:
    def __init__(
        self,
        use_cpu_offload=False,
        n_steps=40,
        high_noise_frac=.8
    ):
        base_name = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_name = "stabilityai/stable-diffusion-xl-refiner-1.0"

        self.base = DiffusionPipeline.from_pretrained(base_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
        self.refiner = DiffusionPipeline.from_pretrained(refiner_name,text_encoder_2=self.base.text_encoder_2,vae=self.base.vae,torch_dtype=torch.float16,use_safetensors=True,variant="fp16").to(device)
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
        if use_cpu_offload:
            self.base.enable_model_cpu_offload()
            self.refiner.enable_model_cpu_offload()
        self.n_steps = n_steps
        self.high_noise_frac = high_noise_frac

    def generate(self, prompts):
        images = self.base(
            prompts,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent"
        ).images
        images = self.refiner(
            prompts,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=images
        ).images
        return images