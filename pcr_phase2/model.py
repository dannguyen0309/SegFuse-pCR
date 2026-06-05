from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PhaseEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, out_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            Conv3DBlock(in_channels, base_channels),
            Conv3DBlock(base_channels, base_channels * 2, stride=2),
            Conv3DBlock(base_channels * 2, out_channels, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConvNormAct3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3D(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + residual)


class MedicalNetResNet3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, out_channels: int) -> None:
        super().__init__()
        stage1_channels = max(base_channels, 32)
        stage2_channels = max(base_channels * 2, 64)
        stage3_channels = max(base_channels * 4, 128)

        self.stem = nn.Sequential(
            ConvNormAct3D(in_channels, stage1_channels, kernel_size=7, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock3D(stage1_channels, stage1_channels),
            ResidualBlock3D(stage1_channels, stage1_channels),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock3D(stage1_channels, stage2_channels, stride=2),
            ResidualBlock3D(stage2_channels, stage2_channels),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock3D(stage2_channels, stage3_channels, stride=2),
            ResidualBlock3D(stage3_channels, stage3_channels),
        )
        self.head = ConvNormAct3D(stage3_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def _resolve_medicalnet_root() -> Path:
    medicalnet_root = Path(__file__).resolve().parents[1] / "third_party" / "MedicalNet"
    if not medicalnet_root.exists():
        raise RuntimeError("Official MedicalNet source not found under third_party/MedicalNet.")
    return medicalnet_root


def _import_official_medicalnet_generate_model():
    medicalnet_root = _resolve_medicalnet_root()
    medicalnet_path = medicalnet_root.as_posix()
    if medicalnet_path not in sys.path:
        sys.path.insert(0, medicalnet_path)
    medicalnet_model = importlib.import_module("model")
    generate_model = getattr(medicalnet_model, "generate_model", None)
    if not callable(generate_model):
        raise RuntimeError("Official MedicalNet source not found under third_party/MedicalNet.")
    return generate_model


def _build_official_medicalnet_backbone(
    in_channels: int,
    model_depth: int,
    resnet_shortcut: str,
) -> nn.Module:
    generate_model = _import_official_medicalnet_generate_model()
    medicalnet_options = SimpleNamespace(
        model="resnet",
        model_depth=model_depth,
        resnet_shortcut=resnet_shortcut,
        input_D=1,
        input_H=1,
        input_W=1,
        n_seg_classes=1,
        no_cuda=True,
        phase="test",
        pretrain_path="",
        gpu_id=[0],
    )
    backbone = generate_model(medicalnet_options)
    if isinstance(backbone, tuple):
        backbone = backbone[0]
    backbone.conv_seg = nn.Identity()
    if int(in_channels) != 1:
        old_conv1 = backbone.conv1
        backbone.conv1 = nn.Conv3d(
            int(in_channels),
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            dilation=old_conv1.dilation,
            bias=old_conv1.bias is not None,
        )
    return backbone


def _infer_medicalnet_feature_dim(backbone: nn.Module) -> int:
    last_block = backbone.layer4[-1]
    if hasattr(last_block, "bn3"):
        return int(last_block.bn3.num_features)
    if hasattr(last_block, "bn2"):
        return int(last_block.bn2.num_features)
    raise RuntimeError("Unable to infer MedicalNet feature dimension from the official backbone.")


def _normalize_medicalnet_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        while True:
            for prefix in ("module.", "model.", "net."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break
            else:
                break
        cleaned[clean_key] = value
    return cleaned


def _load_medicalnet_checkpoint(path: str | Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "net", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _normalize_medicalnet_state_dict(value)
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return _normalize_medicalnet_state_dict(checkpoint)
    if isinstance(checkpoint, dict):
        tensor_like_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
        if tensor_like_items:
            return _normalize_medicalnet_state_dict(tensor_like_items)
    raise RuntimeError(f"Unsupported MedicalNet checkpoint format: {Path(path).as_posix()}")


def _adapt_conv1_weight(checkpoint_weight: torch.Tensor, target_weight: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if checkpoint_weight.shape == target_weight.shape:
        return checkpoint_weight, False
    if checkpoint_weight.ndim != 5 or target_weight.ndim != 5:
        raise RuntimeError(
            f"Cannot adapt conv1 weight from {tuple(checkpoint_weight.shape)} to {tuple(target_weight.shape)}"
        )
    source_channels = int(checkpoint_weight.shape[1])
    target_channels = int(target_weight.shape[1])
    if source_channels == target_channels:
        return checkpoint_weight, False
    if source_channels == 1 and target_channels > 1:
        adapted = checkpoint_weight.repeat(1, target_channels, 1, 1, 1) / float(target_channels)
        return adapted, True
    if target_channels == 1 and source_channels > 1:
        adapted = checkpoint_weight.mean(dim=1, keepdim=True)
        return adapted, True
    raise RuntimeError(
        f"Cannot adapt conv1 weight from {tuple(checkpoint_weight.shape)} to {tuple(target_weight.shape)}"
    )


class OfficialMedicalNetResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        dropout: float,
        pretrained_path: str,
        model_depth: int,
        resnet_shortcut: str,
    ) -> None:
        super().__init__()
        self.pretrained_path = str(pretrained_path)
        self.backbone = _build_official_medicalnet_backbone(
            in_channels=in_channels,
            model_depth=model_depth,
            resnet_shortcut=resnet_shortcut,
        )
        self.feature_dim = _infer_medicalnet_feature_dim(self.backbone)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projector = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
        )
        self.load_report = self._load_pretrained_weights(Path(pretrained_path))

    def _forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def _load_pretrained_weights(self, pretrained_path: Path) -> Dict[str, object]:
        if not pretrained_path.exists():
            raise FileNotFoundError(f"MedicalNet pretrained checkpoint not found: {pretrained_path.as_posix()}")

        checkpoint_state = _load_medicalnet_checkpoint(pretrained_path)
        model_state = self.backbone.state_dict()
        adapted_state: Dict[str, torch.Tensor] = dict(checkpoint_state)
        conv1_loaded = False
        conv1_adapted = False

        if "conv1.weight" in adapted_state and "conv1.weight" in model_state:
            adapted_conv1, did_adapt = _adapt_conv1_weight(adapted_state["conv1.weight"], model_state["conv1.weight"])
            adapted_state["conv1.weight"] = adapted_conv1
            conv1_loaded = True
            conv1_adapted = did_adapt

        load_result = self.backbone.load_state_dict(adapted_state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Official MedicalNet weight loading failed unexpectedly. "
                f"missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}"
            )

        loaded_keys = [key for key in model_state.keys() if key in adapted_state]
        skipped_keys = sorted(key for key in checkpoint_state.keys() if key not in model_state)
        report = {
            "pretrained_path": pretrained_path.as_posix(),
            "total_model_keys": int(len(model_state)),
            "loaded_keys": int(len(loaded_keys)),
            "missing_keys": [],
            "unexpected_keys": [],
            "skipped_keys": skipped_keys,
            "conv1_loaded": bool(conv1_loaded),
            "conv1_adapted": bool(conv1_adapted),
            "conv1_weight_shape": list(model_state["conv1.weight"].shape) if "conv1.weight" in model_state else None,
            "checkpoint_key_count": int(len(checkpoint_state)),
        }
        min_loaded_keys = max(10, int(round(len(model_state) * 0.8)))
        if report["loaded_keys"] < min_loaded_keys:
            raise RuntimeError(
                "Too few pretrained MedicalNet keys were loaded. "
                f"loaded={report['loaded_keys']} total={report['total_model_keys']}"
            )
        return report

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self._forward_feature_map(x)
        pooled = self.pool(feature_map)
        return self.projector(pooled)

    def debug_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self._forward_feature_map(x)
        token = self.projector(self.pool(feature_map))
        return feature_map, token


class OfficialMedicalNetResNet18Encoder(OfficialMedicalNetResNetEncoder):
    def __init__(self, in_channels: int, d_model: int, dropout: float, pretrained_path: str) -> None:
        super().__init__(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=pretrained_path,
            model_depth=18,
            resnet_shortcut="A",
        )


class OfficialMedicalNetResNet34Encoder(OfficialMedicalNetResNetEncoder):
    def __init__(self, in_channels: int, d_model: int, dropout: float, pretrained_path: str) -> None:
        super().__init__(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=pretrained_path,
            model_depth=34,
            resnet_shortcut="A",
        )


class OfficialMedicalNetResNet50Encoder(OfficialMedicalNetResNetEncoder):
    def __init__(self, in_channels: int, d_model: int, dropout: float, pretrained_path: str) -> None:
        super().__init__(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=pretrained_path,
            model_depth=50,
            resnet_shortcut="B",
        )


def _select_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item
            if isinstance(item, (list, tuple, dict)):
                try:
                    return _select_tensor(item)
                except TypeError:
                    continue
    if isinstance(output, dict):
        for item in reversed(list(output.values())):
            if torch.is_tensor(item):
                return item
            if isinstance(item, (list, tuple, dict)):
                try:
                    return _select_tensor(item)
                except TypeError:
                    continue
    raise TypeError("Backbone did not return a tensor-like encoder feature map.")


def build_shared_encoder(
    encoder_type: str,
    in_channels: int,
    base_channels: int,
    out_channels: int,
    d_model: Optional[int] = None,
    dropout: Optional[float] = None,
    medicalnet_pretrained_path: Optional[str] = None,
) -> nn.Module:
    if encoder_type == "official_medicalnet_resnet18":
        if d_model is None:
            raise RuntimeError("official_medicalnet_resnet18 requires d_model to build the projection head.")
        if dropout is None:
            dropout = 0.0
        if not medicalnet_pretrained_path:
            raise RuntimeError("--medicalnet-pretrained-path is required for official_medicalnet_resnet18.")
        return OfficialMedicalNetResNet18Encoder(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=medicalnet_pretrained_path,
        )
    if encoder_type == "official_medicalnet_resnet34":
        if d_model is None:
            raise RuntimeError("official_medicalnet_resnet34 requires d_model to build the projection head.")
        if dropout is None:
            dropout = 0.0
        if not medicalnet_pretrained_path:
            raise RuntimeError("--medicalnet-pretrained-path is required for official_medicalnet_resnet34.")
        return OfficialMedicalNetResNet34Encoder(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=medicalnet_pretrained_path,
        )
    if encoder_type == "official_medicalnet_resnet50":
        if d_model is None:
            raise RuntimeError("official_medicalnet_resnet50 requires d_model to build the projection head.")
        if dropout is None:
            dropout = 0.0
        if not medicalnet_pretrained_path:
            raise RuntimeError("--medicalnet-pretrained-path is required for official_medicalnet_resnet50.")
        return OfficialMedicalNetResNet50Encoder(
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
            pretrained_path=medicalnet_pretrained_path,
        )

    raise ValueError("Unsupported encoder_type. Expected official_medicalnet_resnet18, official_medicalnet_resnet34, or official_medicalnet_resnet50.")


def combine_phase_with_mask(phase: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if phase.ndim != 5:
        raise AssertionError(f"phase must have shape [B, 1, D, H, W], got {tuple(phase.shape)}")
    if mask.ndim != 5:
        raise AssertionError(f"mask must have shape [B, 1, D, H, W], got {tuple(mask.shape)}")
    if phase.shape[0] != mask.shape[0] or phase.shape[2:] != mask.shape[2:]:
        raise AssertionError(
            f"phase and mask must align spatially, got phase={tuple(phase.shape)} mask={tuple(mask.shape)}"
        )
    if phase.shape[1] != 1 or mask.shape[1] != 1:
        raise AssertionError(
            f"phase and mask must each have one channel, got phase={tuple(phase.shape)} mask={tuple(mask.shape)}"
        )
    return torch.cat([phase, mask], dim=1)


class Phase2MultimodalAttentionClassifier(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        dim_feedforward: int = 512,
        encoder_type: str = "official_medicalnet_resnet18",
        encoder_base_channels: int = 32,
        encoder_out_channels: int = 128,
        shared_encoder: bool = True,
        use_image: bool = True,
        use_clinical: bool = True,
        fusion_type: str = "attention",
        use_mask_channel: bool = False,
        return_attention: bool = False,
        medicalnet_pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.use_image = bool(use_image)
        self.use_clinical = bool(use_clinical)
        self.fusion_type = fusion_type
        self.use_mask_channel = bool(use_mask_channel)
        self.return_attention = bool(return_attention)
        if not self.use_image and not self.use_clinical:
            raise ValueError("At least one of use_image or use_clinical must be enabled.")
        if self.fusion_type not in {"attention", "concat"}:
            raise ValueError("fusion_type must be one of: attention, concat")
        if not shared_encoder:
            raise NotImplementedError("Only shared_encoder=True is supported.")

        self.phase_token_count = 3 if self.use_image else 0
        self.clinical_token_count = 1 if self.use_clinical else 0
        self.token_count = self.phase_token_count + self.clinical_token_count

        self.shared_3d_encoder = None
        self.shared_encoder = None
        self.image_encoder_frozen = False
        self.medicalnet_load_report: Dict[str, object] = {"enabled": False, "used": False}
        if self.use_image:
            input_channels = 2 if self.use_mask_channel else 1
            self.shared_3d_encoder = build_shared_encoder(
                encoder_type=encoder_type,
                in_channels=input_channels,
                base_channels=encoder_base_channels,
                out_channels=encoder_out_channels,
                d_model=d_model,
                dropout=dropout,
                medicalnet_pretrained_path=medicalnet_pretrained_path,
            )
            self.shared_encoder = self.shared_3d_encoder
            self.medicalnet_load_report = dict(getattr(self.shared_3d_encoder, "load_report", {"enabled": True}))
            self.medicalnet_load_report["enabled"] = True
            self.medicalnet_load_report["used"] = True

        self.clinical_encoder = None
        if self.use_clinical:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(clinical_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )

        if self.fusion_type == "attention":
            token_slots = max(self.token_count, 1)
            self.token_type_embeddings = nn.Parameter(torch.zeros(token_slots, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fusion_norm = nn.LayerNorm(d_model)
            classifier_input_dim = d_model
        else:
            self.token_type_embeddings = None
            self.transformer = None
            self.fusion_norm = nn.LayerNorm(d_model * self.token_count)
            classifier_input_dim = d_model * self.token_count

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def set_image_encoder_frozen(self, frozen: bool) -> None:
        self.image_encoder_frozen = bool(frozen)
        if self.shared_3d_encoder is None:
            return
        for parameter in self.shared_3d_encoder.parameters():
            parameter.requires_grad = not frozen
        if frozen:
            self.shared_3d_encoder.eval()

    def train(self, mode: bool = True) -> "Phase2MultimodalAttentionClassifier":
        super().train(mode)
        if mode and self.image_encoder_frozen and self.shared_3d_encoder is not None:
            self.shared_3d_encoder.eval()
        return self

    def _encode_phase(self, phase_input: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.shared_3d_encoder is None:
            raise RuntimeError("Image pathway is disabled, so phase encoding is unavailable.")
        encoder_input = phase_input
        if self.use_mask_channel:
            if mask is None:
                raise ValueError("mask must be provided when use_mask_channel=True")
            encoder_input = combine_phase_with_mask(phase_input, mask)
        return self.shared_3d_encoder(encoder_input)

    def _encode_tokens(
        self,
        image: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        clinical: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        tokens: List[torch.Tensor] = []
        if self.use_image:
            if image is None:
                raise ValueError("image tensor is required when use_image=True")
            if image.ndim != 5:
                raise AssertionError(f"image must have shape [B, 3, D, H, W], got {tuple(image.shape)}")
            if image.shape[1] != 3:
                raise AssertionError(f"image must contain exactly 3 DCE phases, got shape {tuple(image.shape)}")
            phase_inputs = [image[:, 0:1], image[:, 1:2], image[:, 2:3]]
            for phase_input in phase_inputs:
                tokens.append(self._encode_phase(phase_input, mask))

        if self.use_clinical:
            if clinical is None:
                raise ValueError("clinical tensor is required when use_clinical=True")
            if clinical.ndim != 2:
                raise AssertionError(f"clinical must have shape [B, F], got {tuple(clinical.shape)}")
            assert self.clinical_encoder is not None
            tokens.append(self.clinical_encoder(clinical))

        if not tokens:
            raise ValueError("No tokens were produced by the configured modalities.")
        return tokens

    def forward(
        self,
        image: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        clinical: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        tokens = self._encode_tokens(image, mask, clinical)

        if self.fusion_type == "attention":
            token_tensor = torch.stack(tokens, dim=1)
            token_tensor = token_tensor + self.token_type_embeddings[: token_tensor.shape[1]].unsqueeze(0)
            encoded = self.transformer(token_tensor) if self.transformer is not None else token_tensor
            fusion = self.fusion_norm(encoded.mean(dim=1))
            output_tokens = encoded
        else:
            token_tensor = torch.cat(tokens, dim=1)
            fusion = self.fusion_norm(token_tensor)
            output_tokens = token_tensor.unsqueeze(1)

        logits = self.classifier(fusion).squeeze(1)
        if logits.ndim != 1:
            raise AssertionError(f"logits must have shape [B], got {tuple(logits.shape)}")

        output = {"logits": logits, "fusion": fusion, "tokens": output_tokens}
        if self.return_attention:
            output["attention_weights"] = torch.empty(0, device=logits.device)
        return output

    def debug_forward(
        self,
        image: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        clinical: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        phase_feature_maps: List[torch.Tensor] = []
        phase_tokens: List[torch.Tensor] = []
        tokens: List[torch.Tensor] = []

        if self.use_image:
            if image is None:
                raise ValueError("image tensor is required when use_image=True")
            phase_inputs = [image[:, 0:1], image[:, 1:2], image[:, 2:3]]
            for phase_input in phase_inputs:
                encoder_input = phase_input
                if self.use_mask_channel:
                    if mask is None:
                        raise ValueError("mask must be provided when use_mask_channel=True")
                    encoder_input = combine_phase_with_mask(phase_input, mask)
                feature_map, phase_token = self.shared_3d_encoder.debug_forward(encoder_input)
                phase_feature_maps.append(feature_map)
                phase_tokens.append(phase_token)
                tokens.append(phase_token)

        if self.use_clinical:
            if clinical is None:
                raise ValueError("clinical tensor is required when use_clinical=True")
            assert self.clinical_encoder is not None
            clinical_token = self.clinical_encoder(clinical)
            tokens.append(clinical_token)
        else:
            clinical_token = None

        if self.fusion_type == "attention":
            token_tensor = torch.stack(tokens, dim=1)
            token_tensor = token_tensor + self.token_type_embeddings[: token_tensor.shape[1]].unsqueeze(0)
            encoded = self.transformer(token_tensor) if self.transformer is not None else token_tensor
            fusion = self.fusion_norm(encoded.mean(dim=1))
            output_tokens = encoded
        else:
            token_tensor = torch.cat(tokens, dim=1)
            fusion = self.fusion_norm(token_tensor)
            output_tokens = token_tensor.unsqueeze(1)

        logits = self.classifier(fusion).squeeze(1)
        return {
            "phase_feature_maps": phase_feature_maps,
            "phase_tokens": phase_tokens,
            "clinical_token": clinical_token if clinical_token is not None else torch.empty(0, device=logits.device),
            "final_tokens": output_tokens,
            "logits": logits,
            "fusion": fusion,
        }


Phase2MaskGuidedTransformerClassifier = Phase2MultimodalAttentionClassifier

