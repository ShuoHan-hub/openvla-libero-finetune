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
# LIBERO Evaluation Debug Notes

## 🧠 Overview

This note summarizes key issues and fixes when evaluating a LoRA fine-tuned OpenVLA model on LIBERO.

> Core takeaway:
> **OpenVLA evaluation requires more than just loading a checkpoint — LoRA, processor, and dataset statistics must all be correctly integrated.**

---

## ⚙️ Setup

* Base model: `openvla-7b`
* Fine-tuning: LoRA (`peft`)
* Evaluation: LIBERO benchmark (`libero_spatial`)
* Environment: MuJoCo + robosuite + GPU cluster

---

## 🚨 Issues & Fixes

### 1. LoRA Not Loaded

**Problem**

Evaluation script only loads base model:

```bash
--pretrained_checkpoint openvla-7b
```

➡️ Result: running baseline instead of fine-tuned model.

**Fix**

```python
from peft import PeftModel

model = PeftModel.from_pretrained(model, lora_path)
```

---

### 2. LoRA Not Merged for Inference

**Problem**

LoRA adapter loaded but not merged → suboptimal inference.

**Fix**

```python
model = model.merge_and_unload()
```

---

### 3. Missing `dataset_statistics.json`

**Error**

```
Action un-norm key libero_spatial not found
```

**Reason**

OpenVLA outputs **normalized actions**, requiring dataset statistics to recover real actions.

**Fix**

```python
import json

with open(stats_path, "r") as f:
    stats = json.load(f)

model.norm_stats = {
    cfg.task_suite_name: stats
}
```

---

### 4. Incorrect `norm_stats` Format

**Wrong**

```python
model.norm_stats = stats
```

**Correct**

```python
model.norm_stats = {
    "libero_spatial": stats
}
```

➡️ OpenVLA expects a mapping: `task_name → stats`

---

### 5. Merged Checkpoint Breaks Processor

**Error**

```
Unrecognized processing class
```

**Reason**

Merged checkpoints may lose HuggingFace processor/tokenizer metadata.

**Fix**

* Always load processor from **base model**

```python
processor = get_processor(cfg)
```

* Do NOT rely on merged checkpoint for processor

---

### 6. Eval Runs but No Rollout

**Symptom**

```
Task suite: libero_spatial
(no further output)
```

**Possible Causes**

* Environment initialization stuck (`get_libero_env`)
* MuJoCo / EGL issues
* Silent crash or tqdm not refreshing

**Debug Strategy**

```python
print(">>> before env init")
env, task_description = get_libero_env(...)
print(">>> after env init")
```

---

## 🧩 Final Correct Pipeline

```
openvla-7b (base)
        ↓
+ LoRA adapter (PeftModel)
        ↓
merge_and_unload()
        ↓
+ dataset_statistics.json (norm_stats)
        ↓
+ processor (from base model)
        ↓
→ LIBERO rollout
```

---

## ✅ Minimal Working Example

```python
# Load base model
model = get_model(cfg)

# Load LoRA
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# Load stats
with open(stats_path, "r") as f:
    stats = json.load(f)

model.norm_stats = {
    cfg.task_suite_name: stats
}

# Load processor
processor = get_processor(cfg)
```

---

## 🔍 Key Insights

### 1. OpenVLA is not a standalone model

It depends on:

* processor
* dataset normalization
* environment interface

---

### 2. LoRA is only part of the system

```
Model = Base + LoRA + Stats + Processor
```

---

### 3. Evaluation = System Test

Not just inference:

```
model output → action → unnormalize → env.step()
```

---

## 🚀 Next Steps

* Compare **baseline vs LoRA** success rate
* Evaluate generalization on:

  * `libero_object`
  * `libero_goal`
  * `libero_10 / libero_90`
* Log results (W&B / CSV)

---

## 🧠 Final Takeaway

> Debugging OpenVLA is not just fixing code —
> it's aligning **model, data, and environment** into a consistent pipeline.

## 🧠 OpenVLA Finetune Debug Note (LIBERO)

### ❗ Problem

During finetuning, the model completely ignored language instructions and produced identical trajectories for all tasks.

After successfully training with the initially modified `finetune.py`, I found that the success rate during evaluation was still 0. Upon further inspection, I observed that the robot executed the same trajectory for all tasks. I later identified that the issue was due to a misalignment in the language instruction pipeline between training and evaluation. The following describes the debugging and fixing process.

---

### 🔍 Root Causes

1. **Fake language instructions**

   ```python
   instruction = Path(path).stem.replace("_", " ")
   ```

   * Not real task descriptions
   * Highly repetitive → model learns constant policy

2. **Missing `get_prompt_builder`**

   * Current `prismatic` version does NOT include this API
   * Caused import errors and inconsistent input formatting

3. **Dataset mismatch**

   ```python
   instruction = demo["task"]["language_instruction"]
   ```

   * ❌ LIBERO dataset does NOT contain this field
   * Leads to `KeyError: 'task'`

4. **Train / Eval misalignment**

   * Train and eval used different input formats
   * Model learned something unusable at inference

---

### ✅ Final Fix

#### 1. Construct instruction manually (aligned with eval)

```python
instruction = Path(path).stem.replace("_demo", "").replace("_", " ")
```

#### 2. Remove prompt builder (critical)

```python
prompt_builder_fn = None
```

#### 3. Use correct RLDS format

```python
rlds_batch = {
    "dataset_name": "libero_spatial",
    "observation": {
        "image_primary": image[None],
    },
    "action": action[None],
    "task": {
        "language_instruction": instruction,
    },
}
```

#### 4. Verify input (debug)

```python
print(processor.tokenizer.decode(batch["input_ids"][0]))
```

---

### ⚖️ Alignment Check

| Component      | Train           | Eval    | Status |
| -------------- | --------------- | ------- | ------ |
| Image          | ✔               | ✔       | ✅      |
| Language       | ✔ (constructed) | ✔ (env) | ✅      |
| Prompt Builder | ❌               | ❌       | ✅      |
| Tokenization   | ✔               | ✔       | ✅      |

→ ✅ Fully aligned

---

### 🚨 Key Insight

> If language is broken, OpenVLA degenerates into a **vision-only behavior cloning policy**.

---

### 🚀 Next Steps

* Lower LR: `5e-4 → 1e-4`
* Validate task-specific behavior (different instructions → different actions)
* Ensure decoded input actually contains language

---

### 🧩 Takeaway

> OpenVLA = **Vision + Language + Action**
>
> If language pipeline fails → model collapses silently.
