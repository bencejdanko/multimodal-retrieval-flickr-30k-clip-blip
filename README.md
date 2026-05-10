# Multimodal Retrieval: Using Flickr30K to Fine-Tune CLIP and BLIP

We train and evaluate CLIP and BLIP to retrieve relevant images and generate meaningful explanations. We evaluate model behavior using appropriate metrics, identify and diagnose failure cases, improve models through fine-tuning strategies, and balance performance with computational cost.

## Dataset

Flickr30K, available on HuggingFace at https://huggingface.co/datasets/nlphuji/flickr30k.

## CLIP Baseline

Use pretrained CLIP to perform text to image retrieval on the fixed Flick30K test set, and get a baseline result. CLIP is available on HuggingFace at [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). 

| Architecture | Precision |
| --- | --- |
| openai/clip-vit-base-patch32 | 16 FP |

| Method | Recall@1 | Recall@5 | MRR | Observation |
| --- | --- | --- | --- | --- |
| Baseline | 21.77% | 41.60% | 0.3155 | Qualitatively, the model performs decently. However, these scores indicate that the model cannot recall precisely the same image from Flickr30k, only 1/5th of the time. When we give 5 recall allowance, it's 2/5. $0.33$ MRR indicates that we average the correct image every 3rd rank. The dataset is slightly noisy, with some captions being ambiguous and up to interpretation, and some being descriptive, but not matching the caption, which would also cause some level of error. |

### Success Cases

| Query | Ground Truth (Left) vs Top 1-CLIP Retreival (Right) |
| --- | --- | 
| A little girl in a pink dress going into a wooden cabin | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/26ccd257-7f82-4225-b60f-10df95523650" /> |
| A man in a blue shirt is standing on a ladder cleaning a win... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/2e8c8e6b-741c-4c37-8f48-064a70188db9" /> |
| A man is sitting on a chair holding a large stuffed animal | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/d8da7832-c3b2-4aed-8b4e-2bd196455e8d" /> |
| Two men in Germany jumping over a rail at the same time with... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/54e3f5a8-ad83-415a-b77f-0cfec1b13bd6" /> |

The model succeeds when captions are clear and unambiguous. They describe colors, scene, and the time of day, for example. This likely gives the model a more clear semantic embedding that cannot be confused with other images.

### 3 Failure Cases

| Query | Ground Truth (Left) vs Top 1-CLIP Retreival (Right) |
| --- | --- |
| A small image used to signify a broken web image link... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/99e64f90-cda4-429b-a758-93f0403ced08" /> |
| The essence of nothingness.. | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/eb089226-02ce-4937-b989-f13979e698c5" /> |
| You must have a great personality, since you chose to conduc... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/d9ae8868-ae60-4120-9886-45e9d69e042a" /> |
| Not appearing within my field of vision | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/599a187f-d514-432e-8faa-1024abd19645" /> |
| The image links are broken | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/ff74034c-1d0a-4907-8a46-009b833f49be" /> |
| The absence of everything | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/0a196c3b-1de2-4ec4-bb6d-6e3f9040d637" /> |

Some images simply had mismatching queries, which also resulted in failed queries. Obviously mismatched or misannotated results were omited. 

Otherwise, failures were caused by queries that were ambiguous and abstract. For example, "the absence of everything" is up to interpretation. The embeddings have no way to clearly distinguish the semantic meaning behind these without extra tuning, and we get more or less random image results.

### Fixed Query Observation

| Query (Caption) | Top-5 Images |
| --- | --- |
| A child playing with a dog in a park | <img width="1570" height="365" alt="image" src="https://github.com/user-attachments/assets/cf1affdc-bd2a-497e-8c0a-025512ef3f0d" /> |
| A person cooking food in a kitchen | <img width="1564" height="427" alt="image" src="https://github.com/user-attachments/assets/e3d2346e-f119-4733-a472-b5b4e961bd78" /> |
| A group of people hiking in the mountains | <img width="1564" height="427" alt="image" src="https://github.com/user-attachments/assets/473824e4-6c44-4e18-872a-1738a76462d6" /> |
| A street scene with cars and pedestrians at night | <img width="1564" height="427" alt="image" src="https://github.com/user-attachments/assets/ac4b3605-81e1-4bfa-b9b4-eeec3327d9e3" /> |
| A person working on a laptop in a coffee shop | <img width="1570" height="427" alt="image" src="https://github.com/user-attachments/assets/37b82ba2-d25c-4414-8ef2-5f4b20e7eab9" /> |


| Query (Caption) | Top-1 Image | Correct (yes/no) | Rank of Correct Image | Observation |
| --- | --- | --- | --- | --- |
| A child playing with a dog in a park | <img width="484" height="414" alt="image" src="https://github.com/user-attachments/assets/0e419731-89ae-4fc7-b114-cc95d5856c60" /> | No | 2 | The top 1 image is incorrect. There is a couple on a lawn with belongings beside them, however, there is no child or dog. The 2nd image yields a correct interpretation, but only 3/5 include a dog and children playing. This is a good opportunity where tuning can improve the result because the query is clear: we need a child playing, a dog present, and park location. |
| A person cooking food in a kitchen | <img width="471" height="527" alt="image" src="https://github.com/user-attachments/assets/41dc76d8-5431-4649-be39-ae8c9b8b09e7" /> | Yes | 1 | All 5 retrieved results are correct. The query is clear, simple and unambiguous, and thus the results are easily satisfied by the base model. |
| A group of people hiking in the mountains | <img width="429" height="527" alt="image" src="https://github.com/user-attachments/assets/2ea98a8e-5e96-4996-84d3-33dfac7c1a20" /> | Yes | 1 | All 5 results are correct and unambiguously correct. We have a hiking group and mountain location, which are clearly semantically separable and thus the base model is satisfactory. |
| A street scene with cars and pedestrians at night | <img width="441" height="569" alt="image" src="https://github.com/user-attachments/assets/bb267f0d-29c6-41a0-a322-a9dd1ebc16e4" /> | No | 5 | The top 3 images featured people in a street, but no cars. Image 4 had only one pedestrian, when the query asks for plural. Only the fifth image had at least 2 or more people and cars in a street scene. The model may have trouble deciphering plural "cars" and "pedestrians", and "street scene" is a bit ambiguous and has potential to be tuned by the model. |
| A person working on a laptop in a coffee shop | <img width="484" height="376" alt="image" src="https://github.com/user-attachments/assets/3b4dbde4-0a0f-4fb2-8992-171d1f84f474" /> | Yes | 1 | The top 1 image appears to be correct, though whether the location is a coffee shop specifically isn't precise. The top 2 images features correctly people working on laptops, but 2/3 images have no laptops - the 3rd one confuses a case with a laptop, and the 5th only has persons. This query is clear, and should be easily tunable by the model. |

## BLIP Baseline

We then use BLIP to generate captions for test images, and compare generated captions with ground-truth captions. BLIP is available on HuggingFace at Salesforce/blip-image-captioning-base.

| Architecture | Precision | Max New Tokens | Decoding |
| --- | --- | --- | --- |
| Salesforce/blip-image-captioning-base | 16fp | 50 | num_beams=1 |

| Method | BLEU-4  | ROUGE-L | METEOR | BERTScore |
| --- | --- | --- | --- | --- |
| Baseline | 0.1975 | 0.4708 | 0.3233 | 0.9251 |

### Fixed Query Results

| Query | CLIP Retrieval | Generated Caption | Ground Truth Caption | Match Quality / Relevance (1-5) | Key Issue |
| --- | --- | --- | --- | --- | --- |
| A child playing with a dog in a park | <img width="484" height="414" alt="image" src="https://github.com/user-attachments/assets/0e419731-89ae-4fc7-b114-cc95d5856c60" /> | a woman laying on the grass | Ground Truth for Index 29364: Person in black shirt lying on top of person in red shirt in the grass. , A girl and a boy in a park kissing in the grass on a sunny day , Two people are laying down and kissing on a grassy lawn. , A young boy and girl cuddle in a grassy field. , A boy lays on top of a girl in a field. | 2/5 - The caption is non-descriptive, and nor does it describe the scene well. There is not only a woman, but she is laying down with somebody. | Non-descriptive / Missing key information |
| A person cooking food in a kitchen | <img width="471" height="527" alt="image" src="https://github.com/user-attachments/assets/41dc76d8-5431-4649-be39-ae8c9b8b09e7" /> | a man in a kitchen | Ground Truth for Index 1562: A chef busily attending to several flaming pots on burners. , A man cooking with fire in like 5 pots at the same time! , Fire is flaming in the skillet of a man in a white coat. , A chef is cooking multiple dishes at the same time. , A man cooking over high flames. | 4/5 - the caption is technically correct. It is not descriptive of the action the man is taking (cooking, holding a pot). | Non-descriptive |
| A group of people hiking in the mountains | <img width="429" height="527" alt="image" src="https://github.com/user-attachments/assets/2ea98a8e-5e96-4996-84d3-33dfac7c1a20" /> | a clear blue sky | Ground Truth for Index 22865: Two women wearing jeans and a man wearing shorts walking up a gravel mountain road carrying backpacks. , Three people are walking up a mountain trail, while one woman is looking at her camera. , Three people are hiking down a long gravel trail with scenic green hills behind them. , Three people, with backpacks hiking along a dirt road through the mountains. , Three hikers a traversing a trail as a young lady inspects her camera. | 3/5 - the caption is technically accurate, but misses the main context entirely, which is a group of people, and in a mountain scene. | Non-descriptive |
| A street scene with cars and pedestrians at night | <img width="441" height="569" alt="image" src="https://github.com/user-attachments/assets/bb267f0d-29c6-41a0-a322-a9dd1ebc16e4" /> | a group of people walking down a street at night | Ground Truth for Index 24600: An older couple walks a narrow, crowded street under orange sodium vapor lamps at night. , An elderly couple is walking through a city block, holding hands. , A man and a woman in a crowd of people on a street at night. , An older couple is walking down the street. , People walking along on a street at night. | 5/5 - Accurate, it describes the subject, location and time. | Aligned and descriptive |
| A person working on a laptop in a coffee shop | <img width="484" height="376" alt="image" src="https://github.com/user-attachments/assets/3b4dbde4-0a0f-4fb2-8992-171d1f84f474" /> | a woman sitting on a chair | Ground Truth for Index 25188: A woman with blond-hair is sitting in a booth with a drink working on her laptop. , As I slave over this assignment, I cautiously click on the answer! , A woman working on her computer in front of a bright yellow wall. , Woman sitting at a table while working on her laptop computer. , A woman in a white shirt working on her laptop. | 3/5 - Technically correct, but is missing the "working" subject or mention of the laptop, or location | Non-descriptive, too general |

## CLIP Fine-tuning

### Linear Probe

The final model is available at [bdanko/clip-flickr30k-linear-probe-finetune](https://huggingface.co/bdanko/clip-flickr30k-linear-probe-finetune).

Training only a newly initialized projection head while keeping the backbones frozen. We freeze the feature extractor so its weights cannot change. This is fast, requires very little memory, and is meant to prevent catastrophic forgetting.

| Hyperparameter | Value |
| --- | --- |
| Precision | 32 FP |
| Batch Size | 2048 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Contrastive Loss |
| Epochs | 10 |
| Learning rate | 1e-4 |
| Embedding dimension | 512 |

<img width="833" height="470" alt="image" src="https://github.com/user-attachments/assets/300318a5-b736-4815-90b8-17fa0d3cd1c5" />

Epoch loss shows smooth exponential decay towards a local minimum. The sharp points in the steps may have been from the last step batches not being the same divisible as the other steps. Since we were training from entirely new weights, there appears some room to train more, but this would likely not be as productive as other strategies.

### Partial Fine-tune

The final model is available at [bdanko/clip-flickr30k-partial-finetune](https://huggingface.co/bdanko/clip-flickr30k-partial-finetune).

We unfreeze and train:

- projection layers (text and vision) 
- logit scale
- final transformer block (layer -1) of the vision encoder
- final transformer block (layer -1) of the text encoder

| Hyperparameter | Value |
| --- | --- |
| Precision | 32FP |
| Batch Size | 2048 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Contrastive Loss |
| Epochs | 5 |
| Learning rate | 5e-5 |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/75e6f890-2487-472c-9ec8-10c65356a5d1" />

We see exponential decay in the loss, though not as dramatic as a linear probe as we are not starting from randomly initialized weights. We likely trained sufficiently as we don't see much loss decrease in the final epochs.

### LoRA

We add trainable adapter matrices into the attention layers.

| Hyperparameter | Value |
| --- | --- |
| Precision | 32FP |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Contrastive Loss |
| Epochs | 5 |
| Learning rate | 5e-5 |
| Rank | 16 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | None |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/229646c5-ea40-4759-9b4b-5f2259e1384d" />

The average epoch loss converges smoothly to a minimum. However, the step loss is very volatile. We used far smaller batches in this strategy (2048 for partial fine tune, 128 for this LoRA), which likely caused major learning differences across each of the steps.

### Full Fine-tune

You can find the model on Huggingface at [bdanko/clip-flickr30k-full-finetune](https://huggingface.co/bdanko/clip-flickr30k-full-finetune).

Unfreezing and continuing training for the entire model.

| Hyperparameter | Value |
| --- | --- |
| Precision | 32FP |
| Batch Size | 128 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Contrastive Loss |
| Epochs | 5 |
| Learning rate | 5e-5 |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/f560c9d7-7fbc-4e99-bf4d-fb95007a5f99" />

Our epoch loss converges, but definitely not to the same degree as the other strategies within the same computational frame (5 epochs). The step loss remains volatile due smaller batch sizes.

### CLIP Fine-Tuning Results

| Method | Recall@1 | Recall@5 | MRR | Observation |
| --- | --- | --- | --- | --- |
| Baseline | 58.68% | 83.44% | 0.6965 | Qualitatively, the model performs decently. However, these scores indicate that the model cannot recall precisely the same image from Flickr30k, only 1/5th of the time. When we give 5 recall allowance, it's 2/5. $0.33$ MRR indicates that we average the correct image every 3rd rank. The dataset is slightly noisy, with some captions being ambiguous and up to interpretation, and some being descriptive, but not matching the caption, which would also cause some level of error. |
| Linear Probe | 39.92 | 70.56 | 0.5368 | We see a dramatic deterioration of results. Attaching a fresh, randomly initialized head likely removed far too much learned knowledge from the original CLIP, and though we trained from scratch on Flickr30K, results were significantly worse then the baseline. |
| Partial Fine-tune | 72.48 | 92.0 | 0.8101 | One of the top performing strategies. Our unfrozen layers we were training (projection, logit scaling, final encoder layers) were highly learnable for our dataset. |
| LoRA | 72.04 | 92.16 | 0.8089 | This approach was also highly learnable, and reached the same performance as a partial fine tune, though we were required to use smaller batches because we needed to calculate fradients at all the injected layers. |
| Full Fine-tune | 66.66 | 89.58 | 0.7677 | Over the same computational time, full fine tuning did not reach the same performance as a partial fine or LoRA adaptation, though it did not complete its convergence. It's likely that with more training it would reach the same performance, though likely not much better. |

Overall, partial fine tuning was the clear superior choice given Flickr30K. It was computationally cheap, as we did not require storing all activations for forward passes as we did in LoRA and for full fine tuning. We could truly freeze most of the model, and the unfrozen modules we chose were highly learnable and tunable to Flickr30K. We could afford larger batch sizes, and the model converged smoothly.

## BLIP Fine Tuning

### Linear Probe

You can find the model on Huggingface at [bdanko/blip-flickr30k-probe](https://huggingface.co/bdanko/blip-flickr30k-probe).

Frozen BLIP backbone with only decoder-side output components trainable.

| Hyperparameter | Value |
| --- | --- |
| Precision | 16BF |
| Batch Size | 512 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Caption CE Loss |
| Epochs | 10 |
| Learning rate | 1e-4 |
| Embedding dimension | 512 |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/b51b9eb1-1f19-4e06-b831-5914d80f22d5" />

### Partial Fine-tune

You can find the model at [bdanko/blip-flickr30k-partial-finetune](https://huggingface.co/bdanko/blip-flickr30k-partial-finetune) on Huggingface.

Unfreeze and train:

- The classifier
- The LM head
- The last text decoder layer
- The last vision encoder layer

layers of the network alongside the projection heads.

| Hyperparameter | Value |
| --- | --- |
| Precision | 16BF |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Caption CE Loss |
| Epochs | 1 |
| Learning rate | 5e-5 |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/66378256-924a-49a5-b8e7-2c3cf874df0e" />

### LoRA

The model is available at [bdanko/blip-flickr30k-lora-finetune](https://huggingface.co/bdanko/blip-flickr30k-lora-finetune)

We add trainable adapter matrices into the attention layers.

| Hyperparameter | Value |
| --- | --- |
| Precision | 16BF |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Caption CE Loss |
| Epochs | 5 |
| Learning rate | 5e-5 |
| Rank | 16 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | None |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/4363394c-68f2-4cb7-9014-1c4359087cf0" />


### Full Fine-tune

The full fine tuned model is available on Huggingface, at [bdanko/blip-flickr30k-full-finetune](https://huggingface.co/bdanko/blip-flickr30k-full-finetune)

Unfreezing and continuing training for the entire model.

| Hyperparameter | Value |
| --- | --- |
| Precision | 16BF |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning rate scheduler | Linear |
| Loss | Caption CE Loss |
| Epochs | 2 |
| Learning rate | 5e-6 |

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/f39c8047-2a33-4901-9049-9a1082e58a31" />


### BLIP Fine-tuning Results

| Method | BLEU-4 | ROUGE-L | METEOR | BERTScore | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 0.1975 | 0.4708 | 0.3233 | 0.9251 | The baseline ended up being the most strongly generalizable model, with better generalization metrics. |
| Linear Probe | 0.2651 | 0.3151 | 0.2500 | 0.9079 | Better word-matching with BLEU, but shows catastrophic forgetting as ROUGE-L and METEOR drop sharply. |
| Partial Fine-tune | 0.2693 | 0.3232 | 0.2647 | 0.9093 | Peak performance for tuned models, with the highest BLEU score, but still catastropically forgets generalizability. |
| LoRA | 0.1576 | 0.3096 | 0.2692 | 0.8978 | The low-rank update (r=16) may have been insufficient, and we saw severe forgetting and performance across all metrics. |
| Full Fine-tune | 0.2371 | 0.3122 | 0.2170 | 0.9035 | More catastrophic forgetting and degrading performance on all metrics. |

All tuned methods appeared to have underperformed. The baseline was the best generalized model by far. BLIP appears to be highly sensitive to catastrophic forgetting. LoRA and full fine tuning thus did not justify their compute in stability, though we could likely try to significantly decrease the learning rates.

## Model Orchestration

We will design a full orchestrated system where:

1. We retrieve the top-3 images using CLIP 
2. Generate a caption for each image using BLIP 
3. Provide a 2–3 sentence explanation justifying relevance 

We will use our best performing tuned models, which were `bdanko/clip-flickr30k-partial-finetune` and `bdanko/blip-flickr30k-partial-finetune`, and compare qualitative results against the baseline.

| Query                                             |   Rank |   Image ID | BLIP Caption                                              | Relevance (1–5)   | Explanation   |
|:--------------------------------------------------|-------:|-----------:|:----------------------------------------------------------|:------------------|:--------------|
| A child playing with a dog in a park              |      1 |        956 | a woman in a red shirt is laying on the grass.            |                   |               |
| A person cooking food in a kitchen                |      1 |        858 | a man in a black shirt is cooking food.                   |                   |               |
| A group of people hiking in the mountains         |      1 |        745 | two people are walking down a dirt road.                  |                   |               |
| A street scene with cars and pedestrians at night |      1 |         50 | a man in a gray shirt is standing on a sidewalk.          |                   |               |
| A person working on a laptop in a coffee shop     |      1 |        592 | a man and a woman are sitting at a table in a restaurant. |                   |               |

## Artifacts

### CLIP Linear Probe Training

<img width="646" height="461" alt="image" src="https://github.com/user-attachments/assets/f8ddd125-2863-4232-84e6-0f9d02fbcb4b" />

### CLIP Partial Fine Tuning

<img width="655" height="161" alt="image" src="https://github.com/user-attachments/assets/549440d3-0d9c-4056-9050-21f87d54119c" />

### CLIP LoRA Training

<img width="649" height="179" alt="image" src="https://github.com/user-attachments/assets/de7b0b3d-e93c-4f55-8f56-b74a1e4a5956" />

### CLIP Full Parameter Training

<img width="656" height="244" alt="image" src="https://github.com/user-attachments/assets/babdcef5-b2c7-4a20-9535-561cc3ff437f" />

### BLIP Probe Training

<img width="659" height="344" alt="image" src="https://github.com/user-attachments/assets/6fc86ec1-5181-4925-9536-5726e9cf528c" />

### BLIP Partial Fine Tuning

<img width="632" height="458" alt="image" src="https://github.com/user-attachments/assets/a83073a1-6e39-45fb-8eee-0b7a17d9b653" />

### BLIP LoRA Training

<img width="606" height="275" alt="image" src="https://github.com/user-attachments/assets/f4d870f9-f6cd-464f-8e3e-e3c6e4175348" />

### BLIP Full Fine Tuning

<img width="650" height="222" alt="image" src="https://github.com/user-attachments/assets/71bbeb7d-0b21-4340-9706-0e457e69c6f1" />



