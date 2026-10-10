"""
finetune.py
-----------
STUB — fine-tuning is intentionally NOT implemented yet.

This file exists to show where a future fine-tuning script plugs in, so
the architecture is visible before any training code is written. The
intended shape, once someone implements it:

  1. Load the prepared dataset:
         from .dataset import load_prepared_dataset
         dataset = load_prepared_dataset()

  2. Load the base pipeline (same default model the live app uses):
         from diffusers import StableDiffusionPipeline
         pipe = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL_ID)

  3. Wrap the UNet for efficient fine-tuning. LoRA (via Hugging Face's
     `peft` library) is the recommended starting point on consumer
     hardware — full fine-tuning of the whole UNet needs far more VRAM
     than a laptop GPU/CPU has. Diffusers ships a reference LoRA
     fine-tuning script worth reading first:
     https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

  4. Run a standard diffusion training loop: encode each image to a VAE
     latent, sample a random timestep and noise it, predict that noise
     with the UNet conditioned on the caption's CLIP embedding, and
     backprop the MSE between predicted and actual noise.

  5. Save the resulting LoRA weights under `config.CHECKPOINTS_DIR`. A
     fine-tuned checkpoint could later be loaded by backend/model.py as an
     alternative to the base model — e.g. as a second option alongside the
     `MODEL_ID` environment variable — without changing how the app itself
     works.

None of this runs today: calling main() raises on purpose, so it's obvious
this step hasn't been built yet rather than silently doing nothing.
"""


def main():
    raise NotImplementedError(
        "Fine-tuning is not implemented yet. This project currently only prepares "
        "the dataset (see prepare_dataset.py) — see this file's module docstring "
        "for the intended approach, and backend/training/README.md for status."
    )


if __name__ == "__main__":
    main()
