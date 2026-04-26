# Multimodal Retrieval: Using Flickr30K to Fine-Tune CLIP and BLIP

We train and evaluate CLIP and BLIP to retrieve relevant images and generate meaningful explanations. We evaluate model behavior using appropriate metrics, identify and diagnose failure cases, improve models through fine-tuning strategies, and balance performance with computational cost.

## Dataset

Flickr30K, available on HuggingFace at https://huggingface.co/datasets/nlphuji/flickr30k.

## CLIP Baseline

Use pretrained CLIP to perform text to image retrieval on the fixed Flick30K test set, and get a baseline result. CLIP is available on HuggingFace at openai/clip-vit-base-patch32. 

| Method | Recall@1 | Recall@5 | MRR | Observation |
| --- | --- | --- | --- | --- |
| Baseline | | | | |

We provide at least 3 correct and 3 failure examples.

| Query (Caption) | Top-1 Image | Correct (yes/no) | Rank of Correct Image | Observation |
| --- | --- | --- | --- | --- |
| A child playing with a dog in a park | | | | |
| A person cooking food in a kitchen | | | | |
| A group of people hiking in the mountains | | | | |
| A street scene with cars and pedestrians at night | | | | |
| A person working on a laptop in a coffee shop | | | | |

## BLIP Baseline

We then use BLIP to generate captions for test images, and compare generated captions with ground-truth captions. BLIP is available on HuggingFace at Salesforce/blip-image-captioning-base.

| Method | BLEU-4  | ROUGE-L | METEOR | BERTScore |
| --- | --- | --- | --- | --- |
| Baseline | | | | |

We provide 3 correct and 3 failure examples.

| Query | Generated Caption | Ground Truth Caption | Match Quality  | Key Issue |
| --- | --- | --- | --- | --- |
| A child playing with a dog in a park | | | | |
| A person cooking food in a kitchen | | | | |
| A group of people hiking in the mountains | | | | |
| A street scene with cars and pedestrians at night | | | | |
| A person working on a laptop in a coffee shop | | | | |

## CLIP Fine-Tuning

| Method | Recall@1 | Recall@5 | MRR | Observation |
| --- | --- | --- | --- | --- |
| Baseline | | | | |
| Linear Probe | | | |
| Partial Fine-tune | | | |
| LoRA | | | |
| Full Fine-tune | | | |

## BLIP Results

| Method | BLEU-4  | ROUGE-L | METEOR | BERTScore |
| --- | --- | --- | --- | --- |
| Baseline | | | | |
| Linear Probe | | | | |
| Partial Fine-tune | | | | |
| LoRA | | | | |
| Full Fine-tune | | | | |

We explain:

- Which method works best and why 
- Why some methods fail or overfit 
- Trade-offs between performance, compute, and stability

## Model Orchestration

We will design a full orchestrated system where:

1. We retrieve the top-3 images using CLIP 
2. Generate a caption for each image using BLIP 
3. Provide a 2–3 sentence explanation justifying relevance 

| Query | Rank | Image ID | BLIP Caption | Relevance (1–5) | Explanation |
| --- | --- | --- | --- | --- | --- |
| A child playing with a dog in a park | | | | | |
| A person cooking food in a kitchen | | | | | |
| A group of people hiking in the mountains | | | | | |
| A street scene with cars and pedestrians at night | | | | | |
| A person working on a laptop in a coffee shop | | | | | |
