"""CLIP Baseline: Text-to-Image Retrieval on Flickr30K using Modal."""

import modal
import json

app = modal.App("clip-baseline-flickr30k")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.36,<5",
        "datasets>=2.14,<3",
        "Pillow",
        "tqdm",
    )
)


@app.function(image=image, gpu="T4", timeout=1800)
def evaluate_clip_baseline():
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
    import json as json_mod

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model in FP16 for faster inference
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", dtype=torch.float16
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", use_fast=True
    )
    model.eval()

    # Load dataset - use test split
    print("Loading Flickr30K dataset...")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    print(f"Test set size: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    sample0_caption = ds[0]["caption"]
    print(f"Caption type: {type(sample0_caption)}, value: {repr(sample0_caption)[:300]}")

    # Parse captions: each row = 1 image, caption is a list of 5 strings
    num_images = len(ds)
    captions = []
    caption_to_image_idx = []

    for i in range(num_images):
        cap_field = ds[i]["caption"]
        if isinstance(cap_field, str):
            cap_list = json_mod.loads(cap_field)
        else:
            cap_list = cap_field
        for cap in cap_list:
            captions.append(cap)
            caption_to_image_idx.append(i)

    num_captions = len(captions)
    print(f"Unique images: {num_images}, Total captions: {num_captions}")

    # Encode all images in batches (larger batch for T4 w/ FP16)
    print("Encoding images...")
    batch_size = 256
    all_image_embeds = []
    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)
        batch_images = [ds[j]["image"].convert("RGB") for j in range(start, end)]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.get_image_features(**inputs)
            embeds = out if isinstance(out, torch.Tensor) else out.image_embeds
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_image_embeds.append(embeds.float().cpu())
    image_embeds = torch.cat(all_image_embeds, dim=0)

    # Encode all captions in batches
    print("Encoding captions...")
    all_text_embeds = []
    for start in tqdm(range(0, num_captions, batch_size)):
        end = min(start + batch_size, num_captions)
        batch_texts = captions[start:end]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.get_text_features(**inputs)
            embeds = out if isinstance(out, torch.Tensor) else out.text_embeds
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_text_embeds.append(embeds.float().cpu())
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Compute retrieval metrics — vectorized rank computation
    print("Computing retrieval metrics...")
    caption_to_image_idx_t = torch.tensor(caption_to_image_idx, dtype=torch.long)
    ranks = []
    eval_batch = 512
    for start in tqdm(range(0, num_captions, eval_batch)):
        end = min(start + eval_batch, num_captions)
        sims = text_embeds[start:end] @ image_embeds.T  # (batch, num_images)
        sorted_indices = sims.argsort(dim=-1, descending=True)
        gt = caption_to_image_idx_t[start:end].unsqueeze(1)  # (batch, 1)
        batch_ranks = (sorted_indices == gt).nonzero(as_tuple=True)[1] + 1
        ranks.append(batch_ranks)

    ranks = torch.cat(ranks).numpy()
    recall_at_1 = (ranks <= 1).mean() * 100
    recall_at_5 = (ranks <= 5).mean() * 100
    mrr = (1.0 / ranks).mean()

    results = {
        "recall_at_1": round(float(recall_at_1), 2),
        "recall_at_5": round(float(recall_at_5), 2),
        "mrr": round(float(mrr), 4),
        "num_images": num_images,
        "num_captions": num_captions,
    }

    # Collect examples
    print("\nExample retrievals:")
    np.random.seed(42)
    example_indices = np.random.choice(num_captions, size=min(10, num_captions), replace=False)
    examples = []
    for idx in example_indices:
        gt_img_idx = caption_to_image_idx[idx]
        sims = text_embeds[idx] @ image_embeds.T
        sorted_indices = sims.argsort(descending=True)
        rank = (sorted_indices == gt_img_idx).nonzero(as_tuple=True)[0].item() + 1
        top1_img_idx = sorted_indices[0].item()
        examples.append({
            "caption": captions[idx],
            "correct": rank == 1,
            "rank_of_correct": int(rank),
            "top1_image_dataset_idx": int(top1_img_idx),
            "gt_image_dataset_idx": int(gt_img_idx),
        })
        print(f"  Caption: {captions[idx][:80]}...")
        print(f"    Correct@1: {'yes' if rank == 1 else 'no'}, Rank: {rank}")

    results["examples"] = examples

    print(f"\n=== CLIP Baseline Results ===")
    print(f"Recall@1: {recall_at_1:.2f}%")
    print(f"Recall@5: {recall_at_5:.2f}%")
    print(f"MRR: {mrr:.4f}")

    return results


@app.local_entrypoint()
def main():
    results = evaluate_clip_baseline.remote()
    print("\n=== Final Results (local) ===")
    print(json.dumps(results, indent=2))
    with open("clip_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to clip_baseline_results.json")
