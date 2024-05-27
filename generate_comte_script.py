import os
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel
import torch
from data.dataset_generators.multi_situation import MultiSituationPromptsDatasetGenerator

checkpoints = [
#    200,
#    400,
#    600,
#    800,
    1000,
    1200,
    1400,
]

fromages = os.listdir("/Data/guillaume.coutiere/finetune_data")
fromages = ["COMTÉ"]

dataset_generator = MultiSituationPromptsDatasetGenerator(None, 1, None, 300)

for fromage in fromages:
    for chk in checkpoints:

        prompts = dataset_generator.create_prompts([fromage])
        prompts = prompts[fromage]
        prompts = [prompt["prompt"] for prompt in prompts]

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

        unet = UNet2DConditionModel.from_pretrained(f"/Data/guillaume.coutiere/models/image_generators/custom-sdxl-1-5/{fromage}/checkpoint-{chk}/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(f"/Data/guillaume.coutiere/models/image_generators/custom-sdxl-1-5/{fromage}/checkpoint-{chk}/text_encoder")

        pipeline = DiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", unet=unet, text_encoder=text_encoder, dtype=torch.float16, scheduler=scheduler
                    ).to("cuda")
        
        for i, prompt in enumerate(prompts):
            image = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]
            image.save(f"/Data/guillaume.coutiere/photos_generated/{fromage}/{chk}_{i}_{prompt}.png")
        
        del pipeline
        torch.cuda.empty_cache()

