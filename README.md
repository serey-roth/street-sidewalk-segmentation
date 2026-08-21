# Dual-Pipeline Analysis for Pedestrian Sidewalk Safety with DeepLabV3 Segmentation and Qwen2.5-VL Safety Assessment

## Overview
Pre-trained segmentation models, such as DeepLabV3+, are effective at classifying pixel-level details, but are limited to categories in their original training data. Fine-tuning the models to recognize new categories is a possible solution, though it requires annotated training labels for the target images. Vision-Language Models (VLMs) are effective at identifying "what's in an image" through natural-language processing, but are limited due to a lack of spatial reasoning.

This project combines those two approaches into an analysis pipeline for pedestrian sidewalk safety. First, we run pre-trained semantic segmentation with DeepLabV3+ and then perform VLM-based safety assessment with Qwen2.5-VL models. The goal is to evaluate how accurately general-purpose VLMs examine pedestrian sidewalk quality and to what extent they can complement segmentation models in fine-grained scene understanding. 

For our analysis, we use HuggingFace's `segments/sidewalk-semantic` [dataset](https://huggingface.co/datasets/segments/sidewalk-semantic), which consists of street-level pedestrian sidewalk images from Belgium in the summer of 2021. For our models, we use `DeepLabV3-ResNet50` for semantic segmentation and `Qwen2.5-VL-3B-Instruct` and `Qwen2.5-VL-7B-Instruct` for VLM scoring.

## Findings

### DeepLabV3-segmented classes vs. Pre-labeled classes
We evaluated `DeepLabV3-ResNet50` model, which is trained on COCO/Pascal VOC's 20 general object classes, against the `sidewalk-semantic` dataset labels (34 sidewalk-specific classes e.g. `flat-walk`, `flat-curb`, `flat-crosswalk`).

Across 3 sample images, we found that the model identified only `background` and `car` and none of the sidewalk-relevant classes (such as `flat-curb`, `flat-road`, `flat-parkingdriveway`, `nature-vegetation`, etc). 

<img width="954" height="300" alt="Screenshot 2026-08-20 at 2 13 46 PM" src="https://github.com/user-attachments/assets/eef233ff-f713-490f-8c28-eedceea09ee2" />

### VLM-based safety assessment with Qwen2.5-VL
We evaluated 2 Qwen2.5-VL models (`Qwen2.5-VL-3B-Instruct` and `Qwen2.5-VL-7B-Instruct`) on three scores per image: `perceived_safety`, `lighting_quality`, and `obstructions`, using a structured JSON prompt.

We found that
- Adding spatial reference and defining the walkable surface from the pedestrian POV improved obstruction accuracy and got rid of hallucinated obstructions (e.g. flagging garden wall as blocking the path)
- The 3B model misidentified the orange pole as an obstruction in the first image even though it's faraway in the distance
- The 3B model scored pretty low (2/5) across all 3 sample images, even though pictures are well-lit and bright

This is the final iteration of the prompt:
```
prompt = """
Analyze the given sidewalk image, which is shown from a pedestrian's eye-level view.

The walkable path is specifically the continuous paved/tiled surface extending from the bottom of the image forward. This is what a person would actually step on while walking. Garden beds, grass, decorative borders, walls, and anything beyond the edge of the paved/tiled surface are NOT part of the walkable path, even if visible in the same image.

For "perceived_safety": judge based only on the physical condition of the walkable surface itself — is it flat, even, and unobstructed (safe), or cracked, uneven, cluttered, or narrow (unsafe)? Ignore weather, time of day, and surrounding buildings. Score 5 if the surface is flat, clear, and wide. Score 1 if the surface is badly damaged, blocked, or unwalkable. Base your score on specific visual evidence in the walkable surface, not a general impression of the scene.

For "lighting_quality": judge based only on how much visible light is present ON the walkable surface itself. Look specifically at shadows, brightness, and contrast on the pavement/tiles. Score 5 if the surface is brightly and evenly lit with no significant dark areas. Score 1 if the surface is mostly in shadow, dim, or hard to see clearly. If the image is a bright daytime photo with clear visibility of the pavement texture, this should score 4 or 5 — do not score low simply because the image includes shaded elements elsewhere in the scene.

For "obstructions": only list objects physically resting ON the walkable surface that prevent the person from walking safely. Do not include anything located beyond the edge of the surface, even if it appears close by.

Before answering, briefly note in your reasoning what specific visual evidence (texture, cracks, shadows, brightness) led to each score.

Respond ONLY in this exact JSON format, no other text:
{
  "perceived_safety": <1-5, where 5 is very safe>,
  "lighting_quality": <1-5, where 5 is excellent lighting>,
  "obstructions": ["list", "or", "empty", "list"],
  "sidewalk_present": <true/false>,
  "reasoning": "<one sentence citing specific visual evidence>"
}"""
```

<img width="951" height="467" alt="Screenshot 2026-08-20 at 2 14 09 PM" src="https://github.com/user-attachments/assets/5846497a-3935-4c18-9182-fa15d42fe89a" />

## Limitations

- We used the models (`DeepLabV3-ResNet50`, `Qwen2.5-VL-3B/7B-Instruct`) as-is without training or fine-tuning them on the `sidewalk-semantic` dataset itself.
  
- Both Qwen2.5-VL models were run with 4-bit quantization to speed up inference and fit available GPU memory (Colab T4, 16GB), which may affect output quality/precision compared to full-precision inference.
  
- We assessed the safety of the sidewalk images qualitatively through the VLMs, without validating quantitatively through standard metrics such as mIoU score or IRR (Inter-Rater Reliability).

## Future Work

- Instead of using a generic pre-trained segmentation model such as `DeepLabV3-ResNet50`, we can experiment with a sidewalk-focused segmentation framework like CitySurfaces that was trained on sidewalk images from New York City and Boston. [CitySurfaces: City-Scale Semantic Segmentation of Sidewalk Materials](https://arxiv.org/abs/2201.02260)
  
- As of right now, we run these two pipelines independently, but our findings have shown the VLM's lack of spatial awareness can be supplemented by a segmentation model's fine-grained spatial localization. A future work can integrate detection and segmentation into a single pipeline using tools such as Grounded-SAM that allows to segment images using natural-language. [Grounded-SAM in 2026: Why It Still Matters (Even in the SAM3 Era)?](https://eng-mhasan.medium.com/grounded-sam-in-2026-why-it-still-matters-even-in-the-sam3-era-15315532365a) [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159)
