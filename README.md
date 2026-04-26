# Cinematic Speech as a Stress Test for Facial Animation Systems

**Author:** Alex Argese — MSc Data Science Engineering (EURECOM) · MSc Digital Media Engineering (Politecnico di Torino)  
**Date:** April 2026  
**Type:** Portfolio Research Project

---

## Overview

Speech-driven facial animation systems such as [FaceFormer](https://github.com/EvelynFan/FaceFormer) (CVPR 2022) are trained and evaluated almost exclusively on [VOCASET](https://voca.is.tue.mpg.de/) — a phonetically balanced dataset designed for articulatory coverage, not emotional diversity. This project asks a simple question: **what happens when you give these systems the kind of speech they were never trained on?**

The answer, documented across two complementary modules, is that the gap is larger than expected — both at the level of text and at the level of facial motion.

---

## Key Results

| Finding | Value |
|---|---|
| VOCASET neutral speech (deduplicated, balanced) | **68.5%** |
| Disney Cinematic Corpus neutral speech | **32.9%** |
| Human-model emotion agreement on cinematic dialogue | **31.5%** |
| Sadness in human annotation vs model prediction | **32.9% vs 13.7%** |
| FaceFormer upper-face motion suppression vs GT | **−59%** |
| Upper/lip ratio — VOCASET vs DCC | **0.15 vs 1.21 (8× gap)** |

---

## Project Structure

```
├── paper/
│   ├── cinematic_speech_paper.pdf     # compiled paper (read this first)
│   └── cinematic_speech_paper.tex     # LaTeX source
├── notebooks/
│   ├── module1_nlp_emotion_analysis.ipynb
│   └── module2_facial_motion_analysis.ipynb
├── figures/
│   ├── emotion_distribution_comparison_balanced.png
│   ├── agreement_analysis.png
│   ├── ground_truth_vs_model.png
│   ├── bar_gt_ff.png
│   ├── timeseries.png
│   ├── ratio_comparison.png
│   └── ratio_dist.png
└── README.md
```

---

## Module 1 — NLP Analysis

We compare the emotional distribution of VOCASET against a curated **Disney Cinematic Corpus (DCC)** of 73 dialogues from 7 animated films (*Frozen*, *Tangled*, *Coco*, *Encanto*, *Up*, *Inside Out*, *Moana*), classified using [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

**Findings:**
- VOCASET is dominated by neutral speech (68.5%), with no non-neutral category exceeding 12%
- The DCC shows a balanced affective distribution: anger (17.8%), sadness (13.7%), joy and fear (12.3% each)
- Human-model agreement on the DCC is only 31.5% — the model systematically underestimates sadness (32.9% human vs 13.7% model) and overestimates neutral and anger
- The low agreement is not a classifier failure but evidence of cinematic dialogue's complexity: utterances like *"...Then leave."* (Elsa, *Frozen*) carry emotional weight that is opaque to a model processing text in isolation

---

## Module 2 — CV Analysis

We analyze 3D facial vertex displacements from pre-computed FaceFormer predictions and VOCASET ground truth, sourced from the [kaist-ami/Perceptual-3D-Talking-Head](https://github.com/kaist-ami/Perceptual-3D-Talking-Head) evaluation benchmark (CVPR 2025).

Facial regions are defined **geometrically** using coordinate thresholds on the FLAME neutral pose, avoiding dependence on the non-anatomical FLAME vertex ordering.

**Findings:**
- FaceFormer suppresses lip motion by 35% and upper-face motion by 59% relative to ground truth — a disproportionate attenuation driven by the LVE training objective
- The lip/upper-face ratio rises from 6.68× (GT) to 10.66× (FaceFormer), amplifying the lower-face bias
- The upper/lip motion ratio across DCC clips (mean: 1.21) is **8× higher** than VOCASET (0.15), indicating that cinematic dialogue demands a qualitatively different facial motion profile

---

## Data

**VOCASET:** available at [voca.is.tue.mpg.de](https://voca.is.tue.mpg.de/)

**FaceFormer predictions:** available at [kaist-ami/Perceptual-3D-Talking-Head](https://github.com/kaist-ami/Perceptual-3D-Talking-Head/tree/main/evaluation/data_MTM) (Git LFS — download directly from GitHub and upload to Colab)

**Disney Cinematic Corpus:** constructed manually from official screenplays and transcripts. Full corpus defined inline in `notebooks/module1_nlp_emotion_analysis.ipynb`.

---

## How to Run

**Module 1** — open `notebooks/module1_nlp_emotion_analysis.ipynb` in Google Colab. No data upload required; VOCASET is loaded via the `voca` package and the DCC is defined inline.

**Module 2** — open `notebooks/module2_facial_motion_analysis.ipynb` in Google Colab. Download the two `.npy` files from the kaist-ami repository linked above and upload them to your Colab session when prompted.

Both notebooks require only standard Python packages: `numpy`, `matplotlib`, `pandas`, `transformers`, `plotly`.

---

## References

- Fan et al., *FaceFormer: Speech-Driven 3D Facial Animation with Transformers*, CVPR 2022
- Cudeiro et al., *Capture, Learning, and Synthesis of 3D Speaking Styles*, CVPR 2019
- Xing et al., *CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior*, CVPR 2023
- Daněček et al., *EmoVOCA: Speech-Driven Emotional 3D Talking Heads*, arXiv 2024
- Kim et al., *Perceptually Accurate 3D Talking Head Generation*, CVPR 2025
- Hartmann, *Emotion English DistilRoBERTa-base*, HuggingFace 2022

---

## Citation

If you find this work useful, you can cite it as:

```
@misc{argese2026cinematic,
  author = {Argese, Alex},
  title  = {Cinematic Speech as a Stress Test for Facial Animation Systems},
  year   = {2026},
  url    = {https://github.com/alexargese/cinematic-speech-facial-animation}
}
```
