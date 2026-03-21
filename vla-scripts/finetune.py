import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import draccus
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import h5py
import wandb
import tqdm

from accelerate import PartialState
from peft import LoraConfig, get_peft_model

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor


# =========================
# Config
# =========================
@dataclass
class FinetuneConfig:
    # Path to pretrained OpenVLA checkpoint
    vla_path: str = "/work/sme-hans/openvla-7b"

    # Root directory of LIBERO dataset
    data_root_dir: Path = Path("/work/sme-hans/libero_data/datasets/libero_spatial")

    # Directory to save checkpoints
    run_root_dir: Path = Path("/work/sme-hans/runs")

    # Training hyperparameters
    batch_size: int = 4
    max_steps: int = 5000
    save_steps: int = 1000

    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32

    # IMPORTANT: must be 0 when using h5py (avoid multiprocessing issues)
    num_workers: int = 0


# =========================
# Dataset (Final Stable Version)
# =========================
class LiberoDataset(Dataset):
    def __init__(self, data_root_dir, transform=None):
        self.transform = transform
        self.samples = []

        data_root_dir = Path(data_root_dir)
        files = list(data_root_dir.glob("*.hdf5"))

        print(f"[Dataset] found {len(files)} files")

        # Build index of (file, demo, timestep)
        for path in files:
            with h5py.File(path, "r") as f:
                for demo_name in f["data"].keys():
                    T = len(f["data"][demo_name]["actions"])
                    for t in range(T):
                        self.samples.append((path, demo_name, t))

        print(f"[Dataset] total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, demo_name, t = self.samples[idx]

        # Load one timestep from HDF5
        with h5py.File(path, "r") as f:
            demo = f["data"][demo_name]

            image = demo["obs"]["agentview_rgb"][t]
            action = demo["actions"][t]

        # Ensure correct dtype for image (required by vision backbone)
        image = image.astype(np.uint8)

        # Construct language instruction from filename
        # Example: pick_up_the_black_bowl_... → "pick up the black bowl ..."
        instruction = Path(path).stem.replace("_demo", "").replace("_", " ")

        # Build pseudo-RLDS format (required by OpenVLA pipeline)
        rlds_batch = {
            "dataset_name": "libero_spatial",
            "observation": {
                "image_primary": image[None],  # add batch dimension
            },
            "action": action[None],  # add batch dimension
            "task": {
                "language_instruction": instruction.encode(),
            },
        }

        if self.transform:
            rlds_batch = self.transform(rlds_batch)

        return rlds_batch


# =========================
# Training (Final Stable Version)
# =========================
@draccus.wrap()
def finetune(cfg: FinetuneConfig):

    print(f"Training on {cfg.data_root_dir}")

    # ---- device setup ----
    state = PartialState()
    device = torch.device(f"cuda:{state.local_process_index}")
    torch.cuda.set_device(device)

    # Enable TF32 for faster matmul on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- register custom OpenVLA classes ----
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # ---- load processor & model ----
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # ---- apply LoRA ----
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)

    model.train()

    # ---- optimizer ----
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate
    )

    # ---- tokenizer & transform ----
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    # ---- dataset ----
    dataset = LiberoDataset(cfg.data_root_dir, transform=batch_transform)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
        ),
    )

    # ---- Weights & Biases ----
    if state.is_main_process:
        wandb.init(project="openvla")

    # ---- training loop ----
    step = 0
    optimizer.zero_grad()
    pbar = tqdm.tqdm(total=cfg.max_steps)

    while step < cfg.max_steps:
        for batch in dataloader:

            # Move all tensors to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # IMPORTANT: fix dtype mismatch (float32 → bfloat16)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )

            loss = output.loss / cfg.grad_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            if step % 10 == 0 and state.is_main_process:
                wandb.log({"loss": loss.item()}, step=step)

            # ---- checkpoint saving ----
            if step % cfg.save_steps == 0 and step > 0:
                if state.is_main_process:
                    save_dir = cfg.run_root_dir / f"step_{step}"
                    os.makedirs(save_dir, exist_ok=True)

                    processor.save_pretrained(save_dir)
                    model.save_pretrained(save_dir)

                    print(f"[SAVE] {save_dir}")

            step += 1
            if step >= cfg.max_steps:
                break

    pbar.close()


if __name__ == "__main__":
    finetune()
