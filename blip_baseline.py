"""BLIP Baseline: Image Captioning on Flickr30K using Modal."""

import modal
import json

app = modal.App("blip-baseline-flickr30k")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.36,<5",
        "datasets>=2.14,<3",
        "Pillow",
        "tqdm",
        "nltk",
        "rouge-score",
        "bert-score",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')\"")
)


@app.function(image=image, gpu="T4", timeout=3600)
def evaluate_blip_baseline():
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from datasets import load_dataset
    from tqdm import tqdm
    import numpy as np
    import json as json_mod
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model in FP16 for faster inference
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", dtype=torch.float16
    ).to(device)
    model.eval()

    # Load dataset
    print("Loading Flickr30K dataset...")
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)
    print(f"Test set size: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    sample0_caption = ds[0]["caption"]
    print(f"Caption type: {type(sample0_caption)}, value: {repr(sample0_caption)[:300]}")

    # Each row = 1 image with a list of captions
    num_images = len(ds)
    all_reference_captions = []
    for i in range(num_images):
        cap_field = ds[i]["caption"]
        if isinstance(cap_field, str):
            cap_list = json_mod.loads(cap_field)
        else:
            cap_list = cap_field
        all_reference_captions.append(cap_list)

    print(f"Unique images: {num_images}")

    # Generate captions with FP16 autocast and larger batches
    generated_captions = []
    print("Generating captions...")
    batch_size = 64
    for start in tqdm(range(0, num_images, batch_size)):
        end = min(start + batch_size, num_images)
        batch_images = [ds[j]["image"].convert("RGB") for j in range(start, end)]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model.generate(**inputs, max_new_tokens=50)
        for j in range(end - start):
            gen_cap = processor.decode(out[j], skip_special_tokens=True).strip()
            generated_captions.append(gen_cap)

    # Compute metrics
    print("Computing metrics...")

    # BLEU-4
    refs_tokenized = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in all_reference_captions]
    hyps_tokenized = [nltk.word_tokenize(gen.lower()) for gen in generated_captions]
    smooth = SmoothingFunction().method1
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized, smoothing_function=smooth)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for gen, refs in zip(generated_captions, all_reference_captions):
        best = max(scorer.score(ref, gen)["rougeL"].fmeasure for ref in refs)
        rouge_scores.append(best)
    rouge_l = np.mean(rouge_scores)

    # METEOR
    meteor_scores = []
    for gen, refs in zip(generated_captions, all_reference_captions):
        gen_tok = nltk.word_tokenize(gen.lower())
        ref_toks = [nltk.word_tokenize(ref.lower()) for ref in refs]
        m = nltk_meteor(ref_toks, gen_tok)
        meteor_scores.append(m)
    meteor = np.mean(meteor_scores)

    # BERTScore - use first reference
    print("Computing BERTScore (this may take a while)...")
    first_refs = [refs[0] for refs in all_reference_captions]
    P, R, F1 = bert_score_fn(generated_captions, first_refs, lang="en", verbose=True, device=device)
    bert_score_val = F1.mean().item()

    results = {
        "bleu4": round(float(bleu4), 4),
        "rouge_l": round(float(rouge_l), 4),
        "meteor": round(float(meteor), 4),
        "bert_score": round(float(bert_score_val), 4),
        "num_images": num_images,
    }

    # Collect examples
    np.random.seed(42)
    example_indices = np.random.choice(num_images, size=min(10, num_images), replace=False)
    examples = []
    for idx in example_indices:
        examples.append({
            "dataset_idx": int(idx),
            "generated_caption": generated_captions[idx],
            "ground_truth_captions": all_reference_captions[idx],
            "rouge_l": round(float(rouge_scores[idx]), 4),
            "meteor": round(float(meteor_scores[idx]), 4),
            "bert_score_f1": round(float(F1[idx].item()), 4),
        })
        print(f"  Generated: {generated_captions[idx]}")
        print(f"  Reference: {all_reference_captions[idx][0][:80]}...")
        print(f"  ROUGE-L: {rouge_scores[idx]:.4f}, METEOR: {meteor_scores[idx]:.4f}")
        print()

    results["examples"] = examples

    print(f"\n=== BLIP Baseline Results ===")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"BERTScore F1: {bert_score_val:.4f}")

    return results


@app.local_entrypoint()
def main():
    results = evaluate_blip_baseline.remote()
    print("\n=== Final Results (local) ===")
    print(json.dumps(results, indent=2))
    with open("blip_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to blip_baseline_results.json")
