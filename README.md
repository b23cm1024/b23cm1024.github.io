# 🔬 FastSAM: Analysis, Novelties & Extensions
### A Deep Learning Course Project — SAM vs FastSAM with Novel Contributions

[![Paper](https://img.shields.io/badge/Paper-arXiv%202306.12156-red?logo=arxiv)](https://arxiv.org/pdf/2306.12156)
[![FastSAM](https://img.shields.io/badge/FastSAM-GitHub-black?logo=github)](https://github.com/CASIA-IVA-Lab/FastSAM)
[![SAM](https://img.shields.io/badge/SAM-Meta%20AI-blue?logo=meta)](https://github.com/facebookresearch/segment-anything)
[![Roboflow](https://img.shields.io/badge/Reference-Roboflow%20Notebook-purple)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-fast-sam.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Paper Reference](#paper-reference)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Novelties & Contributions](#novelties--contributions)
- [Quantitative Results](#quantitative-results)
  - [SDNET2018 — SAM vs FastSAM (CPU)](#sdnet2018--sam-vs-fastsam-cpu)
  - [SDNET2018 — SAM vs FastSAM (GPU)](#sdnet2018--sam-vs-fastsam-gpu)
  - [CrackForest — Zero-Shot Generalization](#crackforest--zero-shot-generalization)
  - [Novelty 1: CLIP Multi-Query Ensemble](#novelty-1-clip-multi-query-ensemble)
  - [Novelty 2: 5-Stage Mask Refinement Pipeline](#novelty-2-5-stage-mask-refinement-pipeline)
  - [Novelty 3: Prompt Fusion](#novelty-3-prompt-fusion)
- [Inference Speed Analysis](#inference-speed-analysis)
- [Visual Results](#visual-results)
- [Installation & Usage](#installation--usage)
- [References](#references)

---

## Overview

This project provides a comprehensive empirical study and extension of the **FastSAM** model (arXiv:2306.12156) — a real-time segmentation alternative to Meta's Segment Anything Model (SAM). We benchmark both models on real-world structural defect datasets and introduce **three novel contributions** that significantly improve FastSAM's segmentation accuracy:

1. **CLIP Multi-Query Text Ensemble** — boosts text-prompt IoU by +52.2 points
2. **5-Stage Mask Refinement Pipeline** — consistently improves all mask types
3. **Prompt Fusion Architecture** — enables instance-level separation of multiple objects that both SAM and FastSAM merge into one mask

The project spans five Google Colab notebooks covering controlled single-image studies, large-scale dataset evaluation (SDNET2018, CrackForest), and novel architectural contributions.

---

## Paper Reference

> **Fast Segment Anything**  
> Xu Zhao, Wenchao Ding, Yongqi An, Yuexian Du, Tao Yu, Min Li, Ming Tang, Jinqiao Wang  
> *arXiv:2306.12156*, June 2023  
> 📄 [https://arxiv.org/pdf/2306.12156](https://arxiv.org/pdf/2306.12156)

FastSAM replaces SAM's heavyweight Vision Transformer (ViT) encoder with a **YOLOv8-seg** convolutional architecture, achieving comparable segmentation quality at a fraction of the compute cost. The original paper reports **50× speedup** over SAM on GPU.

---

## Datasets

| Dataset | Description | Images | Task | Link |
|---------|-------------|--------|------|------|
| **SDNET2018** | Structural Defects Network — concrete crack images | 30 (eval subset) | Crack segmentation | [Kaggle](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images) |
| **CrackForest** | Road surface crack detection with `.mat` GT annotations | 118 | Zero-shot generalization | [GitHub](https://github.com/cuilimeng/CrackForest-dataset) |
| **SA-1B** | Meta's large-scale segmentation dataset used to train SAM | 11M images | Pretraining reference | [Meta AI](https://ai.facebook.com/datasets/segment-anything/) |
| **Roboflow Demo** | Man-with-dog image used for prompt fusion experiments | 1 | Instance separation | [Roboflow Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-fast-sam.ipynb) |

> **Note:** FastSAM was trained on only ~2% of SA-1B (220K vs 11M images). Our experiments probe how this limited training data affects performance on specialized domains like structural crack detection.

---

## Repository Structure

```
Fast_Sam_DL_Project/
│
├── 📓 FAST_SAM_novelty_Mask_Refinement.ipynb      # Main novelty notebook
│   ├── SAM vs FastSAM baseline comparison (all 3 prompts)
│   ├── Inference timing dashboard
│   ├── Multi-image IoU validation (5 diverse scenes)
│   ├──  NOVELTY 1: CLIP Multi-Query Text Ensemble
│   ├──  NOVELTY 2: 5-Stage Mask Refinement Pipeline
│   └── Grand comparison grid + quantitative metrics (9 metrics)
│
├── 📓 FASTSAM_promptfusion.ipynb                   # Novelty 3: Prompt Fusion
│   ├── SAM vs FastSAM standard segmentation
│   └──  NOVELTY 3: Dual-set prompt fusion with edge-aware confidence
│
├── 📓 FASTSAM_Promptfusion_seg1.ipynb              # Prompt fusion (seg1 image variant)
│
├── 📓 SAM_Vs_FASTSAM__on_SDNET_2018_with__cpu_.ipynb  # SDNET2018 evaluation
│   ├── CPU & GPU benchmarking
│   ├── Pseudo-GT mask generation (adaptive threshold + morphology)
│   └── Per-image IoU, Boundary IoU, Dice metrics
│
├── 📓 SAMvsFASTSAM_Crackforestdataset.ipynb        # CrackForest zero-shot eval
│   ├── Zero-shot inference on 118 crack images
│   └── 8-metric evaluation table + visual samples
│
├── 📊 per_image_results_cpu.csv                    # Per-image metrics (CPU run)
├── 📊 per_image_results_gpu.csv                    # Per-image metrics (GPU run)
├── 🖼️ comparison_grid_cpu.png                      # Visual comparison grid (CPU)
├── 🖼️ comparison_grid_Gpu.png                      # Visual comparison grid (GPU)
├── 🖼️ seg1.png                                     # Prompt fusion segmentation output
├── 📄 dataset                                      # Dataset URL reference
└── 📄 barcode.html                                 # Utility HTML file
```

---

## Novelties & Contributions

###  Novelty 1 — CLIP Multi-Query Text Ensemble

**Problem:** FastSAM's text prompt relies on CLIP to post-hoc select from pre-computed masks. A single query has very high variance — `"a dog"` may return IoU=0.175 while `"dog sitting"` returns IoU=0.696 on the exact same image.

**Root Cause:** CLIP scores each mask crop independently with no scene context. YOLOv8's local CNN receptive field misses global semantics. Background textures can outscore the actual target.

**Our Solution:** Run N semantically related queries, score each candidate mask, and select the one with the best coverage signal. This transforms an unreliable CLIP lookup into a robust ensemble.

```
Algorithm:
  1. Define N semantically related text queries (e.g., ["a dog", "dog sitting", "canine animal", ...])
  2. Run FastSAM text prompt for each → N candidate masks
  3. Select mask with best coverage/quality signal
  4. Apply 5-stage refinement to winner
```

**Result:** IoU **0.175 → 0.696** (+52.2 pts) — largest single improvement in the study.

---

###  Novelty 2 — 5-Stage Mask Refinement Pipeline

Applied universally to all FastSAM mask types (Box, Point, Text, Ensemble).

| Stage | Operation | What it Fixes |
|-------|-----------|---------------|
| **A** | Morphological close + open | Fills micro-holes, removes tiny noise blobs |
| **B** | GrabCut re-pass | Pixel-level boundary correction using colour statistics |
| **C** | Canny edge snapping | Aligns mask boundary to real image edges |
| **D** | Small-object removal | Keeps only the dominant segment |
| **E** | Gaussian contour smooth | Reduces jaggedness |

**Result:** Consistent +1.5–1.6 IoU point improvement across **all** mask types.

---

###  Novelty 3 — Prompt Fusion Architecture

**Problem:** Both SAM and FastSAM merge multiple objects in a scene into a single mask when standard prompts are used. Neither model can separately segment a man and a dog that are physically overlapping in an image.

**Our Solution:** A dual-set 2×3 prompt architecture with majority-vote and edge-aware confidence fusion:

- **Set A prompts** (3 points on subject 1, e.g., man) → SAM-A mask
- **Set B prompts** (3 points on subject 2, e.g., dog) → SAM-B mask
- **Fusion** with edge-aware confidence → two non-overlapping instance masks

**Result:** Prompt Fusion is the **only** method capable of producing separate per-instance masks, winning 6/7 evaluation metrics.

---

## Quantitative Results

### SDNET2018 — SAM vs FastSAM (CPU)

Evaluated on 30 concrete crack images from SDNET2018. Pseudo ground-truth generated via adaptive threshold + morphology.

| Metric | SAM (ViT-B) | FastSAM | Winner |
|--------|-------------|---------|--------|
| mIoU ↑ | **0.0350** | 0.0216 | SAM ★ |
| Boundary IoU ↑ | **0.0941** | 0.0236 | SAM ★ |
| Dice Score ↑ | **0.0645** | 0.0413 | SAM ★ |
| Avg Inference ↓ | 120,959 ms | **4,998 ms** | FastSAM ★ |
| Speedup | — | **~24×** | FastSAM ★ |

> **Key finding (CPU):** SAM maintains higher segmentation quality on thin crack structures. FastSAM is ~24× faster but sacrifices boundary precision.

---

### SDNET2018 — SAM vs FastSAM (GPU)

| Metric | SAM (ViT-B) | FastSAM |
|--------|-------------|---------|
| Avg Inference ↓ | ~1,420 ms | **~39 ms** |
| Speedup | — | **~36×** |

> **Note:** On GPU, FastSAM achieves ~36× speedup. The first image has higher FastSAM latency (~8,588 ms) due to model warm-up; subsequent inference drops to ~38 ms.

---

### CrackForest — Zero-Shot Generalization

Evaluated on all **118 CrackForest images** in everything-mode (zero-shot, no fine-tuning). GT loaded from corrected `.mat` files (`seg==2` = crack pixels).

| Metric | SAM (ViT-H) | FastSAM | Winner |
|--------|-------------|---------|--------|
| IoU ↑ | **0.0149** | 0.0104 | SAM ★ Δ30.5% |
| Dice Score ↑ | **0.0292** | 0.0203 | SAM ★ Δ30.5% |
| Precision ↑ | 0.0150 | **0.0161** | FastSAM ★ Δ7.3% |
| Recall ↑ | **0.9230** | 0.5859 | SAM ★ Δ36.5% |
| F1 Score ↑ | **0.0292** | 0.0203 | SAM ★ Δ30.5% |
| Pixel Accuracy ↑ | 0.0232 | **0.4211** | FastSAM ★ Δ94.5% |
| Specificity ↑ | 0.0085 | **0.4206** | FastSAM ★ Δ98.0% |
| Boundary F1 ↑ | **0.0537** | 0.0280 | SAM ★ Δ48.0% |
| Inference (ms) ↓ | 4,134.5 | **147.3** | FastSAM ★ |
| FPS | 0.24 | **6.79** | FastSAM ★ |
| **Speed Advantage** | — | **28.1×** | FastSAM ★ |

> **Key finding:** SAM wins 5/8 quality metrics (better at recall and boundary precision). FastSAM wins 3/8 (better pixel accuracy and specificity because it under-segments and avoids false positives). FastSAM is **28.1× faster**.

---

### Novelty 1: CLIP Multi-Query Ensemble

| Method | IoU | vs Baseline |
|--------|-----|-------------|
| Baseline — `"a dog"` (single query) | 0.1747 | — |
| Majority vote (≥50%) | 0.1747 | +0.00 pts |
| Best single query — `"dog sitting"` | 0.6963 | **+52.16 pts** |
| **Selected Ensemble Mask**  | **0.6963** | **+52.16 pts** |

Per-query breakdown:

| Query | Coverage | IoU |
|-------|----------|-----|
| `"a dog"` | 49.1% | 0.1747 |
| `"dog sitting"` | 17.2% | **0.6963** |
| `"a brown dog"` | 1.8% | 0.0000 |
| `"a dog in a field"` | 49.1% | 0.1747 |
| `"canine animal"` | 49.1% | 0.1747 |
| `"a pet dog"` | 49.1% | 0.1747 |

---

### Novelty 2: 5-Stage Mask Refinement Pipeline

| Mask | Raw IoU | Refined IoU | Change |
|------|---------|-------------|--------|
| FS — Box | 0.6963 | **0.7123** | +1.60 pts ⬆ |
| FS — Point | 0.1747 | **0.1896** | +1.49 pts ⬆ |
| FS — Text | 0.1747 | **0.1903** | +1.55 pts ⬆ |
| FS — Ensemble  | 0.6963 | **0.7123** | +1.60 pts ⬆ |

---



---

### Novelty 3: Prompt Fusion

Evaluated on man-with-dog image. Ground truth built from SAM's high-quality masks for fairness.

| Metric | SAM | FastSAM | **Prompt Fusion ** | Winner |
|--------|-----|---------|---------------------|--------|
| IoU ↑ | 0.3007 | 0.4328 | **0.7738** | PF ★ (+157.3% vs SAM) |
| Dice Score ↑ | 0.4624 | 0.6041 | **0.8725** | PF ★ (+88.7% vs SAM) |
| Precision ↑ | 0.3728 | 0.4395 | **0.8012** | PF ★ (+114.9% vs SAM) |
| Recall ↑ | 0.6087 | **0.9657** | 0.9577 | FastSAM ★ |
| F1 Score ↑ | 0.4624 | 0.6041 | **0.8725** | PF ★ (+88.7% vs SAM) |
| Pixel Accuracy ↑ | 0.6999 | 0.7317 | **0.9407** | PF ★ (+34.4% vs SAM) |
| Boundary F1 ↑ | 0.2720 | 0.5865 | **0.7281** | PF ★ (+167.7% vs SAM) |
| Inference (s) | 0.115 | 0.455 | 0.429 | SAM ★ |

**Prompt Fusion wins: 6/7 metrics.**

Subject-level results (unique to Prompt Fusion — SAM and FastSAM cannot do this):
- **Person IoU: 0.8015**
- **Dog IoU: 0.7667**

---

## Inference Speed Analysis

### Prompt-level Timing (Novelty Notebook, GPU)

| Method | Mean (ms) | Std (ms) |
|--------|-----------|----------|
| SAM — Image Encoding (ViT-B forward) | 964.8 | ±575.8 |
| SAM — Box Prompt Decode | 65.1 | ±106.4 |
| SAM — Point Prompt Decode | 49.8 | ±83.0 |
| **SAM Total (encode + box decode)** | **1,029.9** | — |
| FastSAM — Box Prompt | 123.9 | ±16.2 |
| FastSAM — Point Prompt | 118.1 | ±12.3 |
| FastSAM — Text Prompt (CLIP) | 5,119.7 | ±6,779.4 |
| **FastSAM speedup vs SAM** | **8.3×** | — |
| **CLIP text overhead vs FS-Box** | **+4,995.8 ms** | ⚠️ kills speed advantage |

> **Critical finding:** FastSAM Box/Point is **8.3× faster** than SAM. However, FastSAM's text prompt via CLIP is **30× slower** than FastSAM Box — completely eliminating the speed advantage. Our CLIP Ensemble addresses quality without adding new latency over the CLIP baseline.

---

## Visual Results

| File | Description |
|------|-------------|
| `comparison_grid_cpu.png` | Side-by-side SAM vs FastSAM on SDNET2018 (CPU run, 30 images) |
| `comparison_grid_Gpu.png` | Side-by-side SAM vs FastSAM on SDNET2018 (GPU run, 30 images) |
| `seg1.png` | Prompt Fusion segmentation output — man and dog separated |
| `per_image_results_cpu.csv` | Full per-image IoU, Boundary IoU, Dice, inference time (CPU) |
| `per_image_results_gpu.csv` | Full per-image IoU, Boundary IoU, Dice, inference time (GPU) |

---

## Installation & Usage

All notebooks are designed to run on **Google Colab** with a GPU runtime.

### Quick Start

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/Fast_Sam_DL_Project.git
cd Fast_Sam_DL_Project
```

### Dependencies (auto-installed in each notebook)

```bash
pip install ultralytics                                          # FastSAM (YOLOv8-seg)
pip install git+https://github.com/facebookresearch/segment-anything.git  # SAM
pip install git+https://github.com/ultralytics/CLIP.git         # CLIP for text prompts
pip install opencv-python scipy matplotlib tqdm Pillow scikit-image
```

### Model Weights

| Model | Checkpoint | Size | Auto-downloaded |
|-------|-----------|------|-----------------|
| FastSAM-x | `FastSAM-x.pt` | ~138 MB | ✅ via Ultralytics |
| SAM ViT-B | `sam_vit_b_01ec64.pth` | ~375 MB | ✅ via Ultralytics |
| SAM ViT-H | `sam_vit_h_4b8939.pth` | ~2.4 GB | ✅ in notebook |

### Notebook Guide

| Notebook | Purpose | Recommended Runtime |
|----------|---------|---------------------|
| `FAST_SAM_novelty_Mask_Refinement.ipynb` | **Start here** — main study + all novelties | GPU |
| `FASTSAM_promptfusion.ipynb` | Prompt Fusion novelty | GPU |
| `FASTSAM_Promptfusion_seg1.ipynb` | Prompt Fusion (seg1 variant) | GPU |
| `SAM_Vs_FASTSAM__on_SDNET_2018_with__cpu_.ipynb` | SDNET2018 evaluation | CPU or GPU |
| `SAMvsFASTSAM_Crackforestdataset.ipynb` | CrackForest zero-shot | GPU (recommended) |

---

## Key Conclusions


  SAM > FastSAM boundary quality (ViT global attention captures fine crack edges)
  FastSAM Box/Point >> FastSAM Text (geometry-based selection beats post-hoc CLIP)
  CLIP Ensemble: text IoU 0.175 → 0.696 (+52.2 pts) ← KEY RESULT
  5-Stage Refinement improves every mask consistently (+1.5–1.6 IoU pts)
  Prompt Fusion: IoU 0.774 vs SAM 0.301, enables per-instance separation
  FastSAM is 8–28× faster than SAM depending on hardware and dataset
  FastSAM text (CLIP) is 30× slower than FastSAM Box — avoid raw text prompts
  Best text-prompt alternative: Grounded-SAM or LangSAM for production use

---

## References

1. **FastSAM Paper (Primary Reference)**  
   Zhao, X. et al. *Fast Segment Anything.* arXiv:2306.12156, 2023.  
   [https://arxiv.org/pdf/2306.12156](https://arxiv.org/pdf/2306.12156)

2. **Segment Anything Model (SAM)**  
   Kirillov, A. et al. *Segment Anything.* arXiv:2304.02643, 2023.  
   [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

3. **YOLOv8 / Ultralytics**  
   Jocher, G. et al. *Ultralytics YOLOv8.* 2023.  
   [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

4. **CLIP (Contrastive Language–Image Pretraining)**  
   Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.  
   [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

5. **SDNET2018 Dataset**  
   Dorafshan, S., Thomas, R.J., Maguire, M. *SDNET2018: A concrete crack image dataset for machine learning applications.* Utah State University, 2018.  
   [https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images)

6. **CrackForest Dataset**  
   Shi, Y. et al. *Automatic Road Crack Detection Using Random Structured Forests.*  
   [https://github.com/cuilimeng/CrackForest-dataset](https://github.com/cuilimeng/CrackForest-dataset)

7. **Roboflow FastSAM Tutorial (Reference Implementation)**  
   Roboflow AI. *How to Segment Anything with FastSAM.*  
   [https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-fast-sam.ipynb](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-fast-sam.ipynb)

8. **SA-1B Dataset (SAM Pretraining)**  
   Meta AI Research. *Segment Anything 1 Billion (SA-1B).*  
   [https://ai.facebook.com/datasets/segment-anything/](https://ai.facebook.com/datasets/segment-anything/)

9. **GrabCut Algorithm**  
   Rother, C., Kolmogorov, V., Blake, A. *GrabCut: Interactive foreground extraction using iterated graph cuts.* SIGGRAPH 2004.

10. **FastSAM Official Repository**  
    CASIA-IVA-Lab. *FastSAM: Fast Segment Anything.*  
    [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)

---

<div align="center">

Based on **FastSAM** (arXiv:2306.12156) by Zhao et al., 2023.

</div>
