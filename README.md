# 🚀 OpenVLA Finetuning on LIBERO

Finetuning OpenVLA-7B on the LIBERO dataset with a custom RLDS data pipeline and LoRA adaptation.

---

## 📌 Overview

This project reproduces and extends the training pipeline of OpenVLA by adapting the LIBERO dataset (HDF5 format) into the RLDS format required by the model.

Key contributions include:

* 🔧 Custom dataset pipeline (HDF5 → RLDS)
* 🤖 End-to-end OpenVLA finetuning
* ⚡ LoRA-based efficient training
* 🖥️ Stable training on HPC cluster
* 📊 Integrated Weights & Biases logging

---

## 🧱 Pipeline

```
LIBERO (HDF5)
    ↓
LiberoDataset (custom)
    ↓
RLDSBatchTransform
    ↓
Prompt + Image + Action Tokens
    ↓
OpenVLA (Vision + Language Model)
```

---

## 📂 Dataset Format

Each LIBERO file is structured as:

```
data/
  demo_0/
    ├── actions
    ├── obs/agentview_rgb
    ├── rewards
    ├── states
```

Converted into pseudo-RLDS format:

```python
rlds_batch = {
    "dataset_name": "libero_spatial",
    "observation": {"image_primary": image[None]},
    "action": action[None],
    "task": {"language_instruction": instruction.encode()},
}
```

---

## 🏋️ Training Setup

* Model: OpenVLA-7B
* Dataset: LIBERO Spatial
* Method: LoRA finetuning
* Batch size: 8
* Training steps: 5000

---

## 📈 Results

Training shows stable convergence:

* Loss drops rapidly at early stage (~9 → ~2.5)
* Stabilizes after ~1000 steps
* No divergence observed

---

## ⚠️ Key Challenges

### 1. Dataset format mismatch

LIBERO uses HDF5 while OpenVLA expects RLDS.

✔️ Solution: build custom wrapper

---

### 2. Mixed precision bug (critical)

```
RuntimeError: Input type (float) and bias type (bfloat16)
```

✔️ Fix:

```python
pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
```

---

### 3. h5py multiprocessing issue

✔️ Fix:

```python
num_workers = 0
```

---

## 🧠 Insights

* OpenVLA is highly sensitive to input data format
* RLDS abstraction is powerful but non-trivial to debug
* LoRA enables efficient finetuning even on limited data

---

## 🔮 Future Work

* Multi-step trajectory prediction
* Larger dataset mixtures (OXE)
* Simulation evaluation on LIBERO benchmark

---
