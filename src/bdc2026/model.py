import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType


class Dinov3LoraClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base_model = AutoModel.from_pretrained(
            cfg.model_name,
            token=cfg.hf_token,
        )

        if cfg.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not find hidden_size in model config.")

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        try:
            self.backbone = get_peft_model(base_model, lora_config)
        except ValueError as e:
            print("\nLoRA target module error.")
            print("Current target modules:", cfg.lora_target_modules)
            print("Some Linear module names in the model:")
            shown = 0
            for name, module in base_model.named_modules():
                if isinstance(module, nn.Linear):
                    print(name)
                    shown += 1
                    if shown >= 100:
                        break
            raise e

        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden_size, cfg.num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(features))


def unwrap_model(model):
    """Return the underlying model when wrapped by DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def count_trainable_params(model):
    model = unwrap_model(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Trainable ratio: {100 * trainable / total:.4f}%")


def get_optimizer(model, cfg):
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            lora_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )


def get_trainable_state_dict(model):
    """
    Save only trainable parameters, unwrapped from DataParallel.

    This keeps checkpoints small and loadable by both single-GPU and multi-GPU
    runs because saved keys do not include the DataParallel `module.` prefix.
    """
    model = unwrap_model(model)
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    full_state = model.state_dict()
    return {k: v.detach().cpu() for k, v in full_state.items() if k in trainable_names}
