# Multimodal Retrieval: Using Flickr30K to Fine-Tune CLIP and BLIP

We train and evaluate CLIP and BLIP to retrieve relevant images and generate meaningful explanations. We evaluate model behavior using appropriate metrics, identify and diagnose failure cases, improve models through fine-tuning strategies, and balance performance with computational cost.

## Dataset

Flickr30K, available on HuggingFace at https://huggingface.co/datasets/nlphuji/flickr30k.

## CLIP Baseline

Use pretrained CLIP to perform text to image retrieval on the fixed Flick30K test set, and get a baseline result. CLIP is available on HuggingFace at openai/clip-vit-base-patch32. 

| Architecture | Precision |
| --- | --- |
| openai/clip-vit-base-patch32 | 16fp |

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


### 3 Failure Cases

| Query | Ground Truth (Left) vs Top 1-CLIP Retreival (Right) |
| --- | --- |
| A small image used to signify a broken web image link... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/99e64f90-cda4-429b-a758-93f0403ced08" /> |
| The essence of nothingness.. | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/eb089226-02ce-4937-b989-f13979e698c5" /> |
| You must have a great personality, since you chose to conduc... | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/d9ae8868-ae60-4120-9886-45e9d69e042a" /> |
| Not appearing within my field of vision | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/599a187f-d514-432e-8faa-1024abd19645" /> |
| The image links are broken | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/ff74034c-1d0a-4907-8a46-009b833f49be" /> |
| The absence of everything | <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/0a196c3b-1de2-4ec4-bb6d-6e3f9040d637" /> |

> [!NOTE]
> Some images simply had mismatching queries, which also resulted in failed queries.
> Obviously mismatched or misannotated results were omited


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
| A child playing with a dog in a park | <img width="484" height="414" alt="image" src="https://github.com/user-attachments/assets/0e419731-89ae-4fc7-b114-cc95d5856c60" /> | No | 2 | The top 1 image is incorrect. There is a couple on a lawn with belongings beside them, however, there is no child or dog. The 2nd image yields a correct interpretation, but only 3/5 include a dog and children playing. |
| A person cooking food in a kitchen | <img width="471" height="527" alt="image" src="https://github.com/user-attachments/assets/41dc76d8-5431-4649-be39-ae8c9b8b09e7" /> | Yes | 1 | All 5 retrieved results are correct and unambiguous. |
| A group of people hiking in the mountains | <img width="429" height="527" alt="image" src="https://github.com/user-attachments/assets/2ea98a8e-5e96-4996-84d3-33dfac7c1a20" /> | Yes | 1 | All 5 results are correct and unambiguously correct. |
| A street scene with cars and pedestrians at night | <img width="441" height="569" alt="image" src="https://github.com/user-attachments/assets/bb267f0d-29c6-41a0-a322-a9dd1ebc16e4" /> | No | 5 | The top 3 images featured people in a street, but no cars. Image 4 had only one pedestrian, when the query asks for plural. Only the fifth image had at least 2 or more people and cars in a street scene. |
| A person working on a laptop in a coffee shop | <img width="484" height="376" alt="image" src="https://github.com/user-attachments/assets/3b4dbde4-0a0f-4fb2-8992-171d1f84f474" /> | Yes | 1 | The top 1 image appears to be correct, though whether the location is a coffee shop specifically isn't precise. The top 2 images features correctly people working on laptops, but 2/3 images have no laptops - the 3rd one confuses a case with a laptop, and the 5th only has persons. |

## BLIP Baseline

We then use BLIP to generate captions for test images, and compare generated captions with ground-truth captions. BLIP is available on HuggingFace at Salesforce/blip-image-captioning-base.

| Architecture | Precision | Max New Tokens | Decoding |
| --- | --- | --- | --- |
| Salesforce/blip-image-captioning-base | 16fp | 50 | num_beams=1 |

| Method | BLEU-4  | ROUGE-L | METEOR | BERTScore |
| --- | --- | --- | --- | --- |
| Baseline | 0.1975 | 0.4708 | 0.3233 | 0.9251 |

We provide 3 correct and 3 failure examples.

| Query | CLIP Retrieval | Generated Caption | Ground Truth Caption | Match Quality / Relevance (1-5) | Key Issue |
| --- | --- | --- | --- | --- | --- |
| A child playing with a dog in a park | <img width="484" height="414" alt="image" src="https://github.com/user-attachments/assets/0e419731-89ae-4fc7-b114-cc95d5856c60" /> | a woman laying on the grass | Ground Truth for Index 29364: Person in black shirt lying on top of person in red shirt in the grass. , A girl and a boy in a park kissing in the grass on a sunny day , Two people are laying down and kissing on a grassy lawn. , A young boy and girl cuddle in a grassy field. , A boy lays on top of a girl in a field. | 2/5 - The caption is quite non-descriptive - there is not just a woman, but she is laying down with somebody. The scene described is quite non descriptive | Non-descriptive / Missing key information |
| A person cooking food in a kitchen | <img width="471" height="527" alt="image" src="https://github.com/user-attachments/assets/41dc76d8-5431-4649-be39-ae8c9b8b09e7" /> | a man in a kitchen | Ground Truth for Index 1562: A chef busily attending to several flaming pots on burners. , A man cooking with fire in like 5 pots at the same time! , Fire is flaming in the skillet of a man in a white coat. , A chef is cooking multiple dishes at the same time. , A man cooking over high flames. | 4/5 - the caption is correct, but non descriptive of the persons or kitchen scene. | Non-descriptive |
| A group of people hiking in the mountains | <img width="429" height="527" alt="image" src="https://github.com/user-attachments/assets/2ea98a8e-5e96-4996-84d3-33dfac7c1a20" /> | a clear blue sky | Ground Truth for Index 22865: Two women wearing jeans and a man wearing shorts walking up a gravel mountain road carrying backpacks. , Three people are walking up a mountain trail, while one woman is looking at her camera. , Three people are hiking down a long gravel trail with scenic green hills behind them. , Three people, with backpacks hiking along a dirt road through the mountains. , Three hikers a traversing a trail as a young lady inspects her camera. | 4/5 - the caption is accurate, a group of people hiking in the mountains, but it's not very descriptive | Non-descriptive |
| A street scene with cars and pedestrians at night | <img width="441" height="569" alt="image" src="https://github.com/user-attachments/assets/bb267f0d-29c6-41a0-a322-a9dd1ebc16e4" /> | a group of people walking down a street at night | Ground Truth for Index 24600: An older couple walks a narrow, crowded street under orange sodium vapor lamps at night. , An elderly couple is walking through a city block, holding hands. , A man and a woman in a crowd of people on a street at night. , An older couple is walking down the street. , People walking along on a street at night. | 4/5 - Accurate, it describes the subject, location and time, but could use more descriptive alignment like the ground truth | Non-descriptive |
| A person working on a laptop in a coffee shop | <img width="484" height="376" alt="image" src="https://github.com/user-attachments/assets/3b4dbde4-0a0f-4fb2-8992-171d1f84f474" /> | a woman sitting on a chair | Ground Truth for Index 25188: A woman with blond-hair is sitting in a booth with a drink working on her laptop. , As I slave over this assignment, I cautiously click on the answer! , A woman working on her computer in front of a bright yellow wall. , Woman sitting at a table while working on her laptop computer. , A woman in a white shirt working on her laptop. | 3/5 - Technically correct, but is missing the "working" subject or mention of the laptop, or location | Non-descriptive, too general |

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
