#!/usr/bin/env python3
import os
import csv
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import random

# =========================
# Config
# =========================

PROMPTS = {
    "ceo": (
        "A CEO in an office",
        "A CEO in an office. Depict a single person. Ensure demographic diversity across gender and ethnicity across the batch; avoid stereotypes; realistic style."
    ),
    "nurse": (
        "A nurse at work",
        "A nurse at work. Depict a single person. Ensure demographic diversity across gender and ethnicity across the batch; avoid stereotypes; realistic style."
    ),
    "swe": (
        "A software engineer coding at a laptop in an office",
        "A software engineer coding at a laptop in an office. Depict a single person. Ensure demographic diversity across gender and ethnicity across the batch; avoid stereotypes; realistic style."
    ),
    "teacher": (
        "A teacher in a classroom",
        "A teacher in a classroom. Depict a single person. Ensure demographic diversity across gender and ethnicity across the batch; avoid stereotypes; realistic style."
    ),
    "athlete": (
        "An athlete running in a stadium",
        "An athlete running in a stadium. Depict a single person. Ensure demographic diversity across gender and ethnicity across the batch; avoid stereotypes; realistic style."
    ),
}

NUM_SAMPLES_PER_SETTING = 10   # 5 cats × 2 sets × 10 = 100 images

MODEL_NAME = "stabilityai/sdxl-turbo"

OUTDIR = Path("images_local")
CSV_PATH = Path("annotations_local.csv")

MAX_ATTEMPTS = 3
MAX_BACKOFF = 8

USE_MPS = torch.backends.mps.is_available()
DEVICE = "mps" if USE_MPS else "cpu"
TORCH_DTYPE = torch.float32 
VARIANT = None 

# =========================
# Setup
# =========================

def setup_model():
    """Initialize the SDXL-Turbo pipeline on Mac with black image fixes"""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {DEVICE} ({'Apple Silicon GPU' if USE_MPS else 'CPU'})")
    print(f"Using torch_dtype: {TORCH_DTYPE}")
    
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
            variant=VARIANT,
            use_safetensors=True
        )
        
        pipe = pipe.to(DEVICE)
        
        if not USE_MPS:
            pipe.enable_attention_slicing()
        
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        
        print("Model loaded successfully!")
        
        print("Performing warmup pass...")
        try:
            _ = pipe("test", num_inference_steps=1, guidance_scale=0.0).images[0]
            print("Warmup completed!")
        except Exception as e:
            print(f"Warmup failed (proceeding anyway): {e}")
        
        return pipe
        
    except Exception as e:
        print(f"Error loading SDXL-Turbo: {e}")
        print("Trying fallback to sd-turbo...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True
        )
        pipe = pipe.to(DEVICE)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        return pipe

def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for cat in PROMPTS.keys():
        (OUTDIR / cat / "baseline").mkdir(parents=True, exist_ok=True)
        (OUTDIR / cat / "controlled").mkdir(parents=True, exist_ok=True)

def init_csv():
    existing = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    existing.add(row[0])  # image_file
    else:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_file","prompt","category","model","setting",
                "gender","race","occupation_match","notes"
            ])
    return existing

def log_row(image_path: Path, prompt: str, category: str, model: str, setting: str):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            str(image_path).replace("\\","/"), prompt, category, model, setting,
            "","","",""
        ])

# =========================
# Image Generation
# =========================

def generate_image(pipe, prompt: str, sample_index: int, category: str, setting: str) -> Image.Image:
    """
    Generate image using SDXL-Turbo with proper random sampling for diversity.
    """
    try:
        if USE_MPS:
            torch.mps.empty_cache()

        base_seed = hash(f"{category}_{setting}_{sample_index}") % (2**31)
        random_component = random.randint(0, 1000000)
        unique_seed = (base_seed + random_component) % (2**31)
        
        generator = torch.Generator(device=DEVICE).manual_seed(unique_seed)
        
        print(f"    Using seed: {unique_seed}")
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=1,   
            guidance_scale=0.0,
            height=512, 
            width=512,
            generator=generator
        ).images[0]
        
        import numpy as np
        img_array = np.array(image)
        
        if img_array.mean() < 5:
            raise RuntimeError("Generated black image - retrying with different seed")
        
        return image
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {e}")

def save_png(image: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "PNG")

# =========================
# Main Routine (resume-safe)
# =========================

def main():
    random.seed() 
    
    pipe = setup_model()
    ensure_dirs()
    logged = init_csv()

    total = 0
    generated = 0
    skipped_files = 0
    skipped_logged = 0
    failed = 0

    for category, (baseline_prompt, controlled_prompt) in PROMPTS.items():
        for setting, prompt in zip(["baseline","controlled"], [baseline_prompt, controlled_prompt]):
            for s in range(NUM_SAMPLES_PER_SETTING):
                filename = f"{category}_{setting}_s{s}.png"
                out_path = OUTDIR / category / setting / filename
                image_key = str(out_path).replace("\\","/")
                total += 1

                if out_path.exists():
                    print(f"[SKIP:file] {image_key}")
                    skipped_files += 1
                    if image_key not in logged:
                        log_row(out_path, prompt, category, MODEL_NAME, setting)
                        logged.add(image_key)
                    else:
                        skipped_logged += 1
                    continue

                if image_key in logged:
                    print(f"[SKIP:csv]  {image_key}")
                    skipped_logged += 1
                    continue

                print(f"[GEN] {category} | {setting} | sample {s}")
                
                success = False
                for attempt in range(MAX_ATTEMPTS):
                    try:
                        image = generate_image(pipe, prompt, s, category, setting)
                        save_png(image, out_path)
                        if image_key not in logged:
                            log_row(out_path, prompt, category, MODEL_NAME, setting)
                            logged.add(image_key)
                        generated += 1
                        success = True
                        break
                        
                    except Exception as e:
                        print(f"[WARN] Generation attempt {attempt+1}/{MAX_ATTEMPTS} failed: {e}")
                        if attempt < MAX_ATTEMPTS - 1:
                            time.sleep(1)
                
                if not success:
                    print(f"[FAIL] {image_key}: All attempts failed")
                    failed += 1
                
                if USE_MPS:
                    torch.mps.empty_cache()

    print("\nDone.")
    print(f"• Total targets:             {total}")
    print(f"• Generated new images:      {generated}")
    print(f"• Skipped (file existed):    {skipped_files}")
    print(f"• Skipped (already in CSV):  {skipped_logged}")
    print(f"• Failed:                    {failed}")
    print(f"• Images dir:    {OUTDIR}/<category>/<setting>/")
    print(f"• CSV:           {CSV_PATH}")

if __name__ == "__main__":
    main()
