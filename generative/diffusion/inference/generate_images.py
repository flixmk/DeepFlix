from torch import autocast
import diffusers
import transformers
import os
from tqdm.notebook import tqdm
from huggingface_hub import notebook_login, login
from diffusers.utils import logging


CLASSES = ['NORMAL', 'CNV', 'DRUSEN', 'DME']


def generate(pipeline, 
                    num_samples,
                    height: int = 512, 
                    width: int = 512, 
                    num_inference_steps: int = 50, 
                    guidance_scale: float = 7.5, 
                    negative_prompt: str = None, 
                    num_images_per_prompt: int = 2,
                    save_path: str = None,):

    iterations = num_samples // num_images_per_prompt
    print(f"Iterations per class: {iterations}")

    for class_prompt in CLASSES:
      print(f"Current class(prompt): {class_prompt}")
      # create fodler structure
      if save_path is not None:
        save_path_class = save_path + f"/{class_prompt}"
        isExist = os.path.exists(save_path_class)
        if not isExist:
          os.makedirs(save_path_class)
      if negative_prompt == "reverse":
        negative_prompt = [x for x in CLASSES if x!=class_prompt]
      for it in range(iterations):
        with autocast("cuda"):
          images = pipeline(
              prompt = class_prompt,
              height = height,
              width = width,
              num_inference_steps = num_inference_steps,
              guidance_scale = guidance_scale,
              negative_prompt = negative_prompt,
              num_images_per_prompt = num_images_per_prompt,
              ).images
        for idx, image in enumerate(images):
          id_num = idx + (it * num_images_per_prompt)
          id = str(id_num).zfill(len(str(num_samples)))
          image.save(f"{save_path_class}/{class_prompt}-({id}).png")

      negative_prompt = "reverse"