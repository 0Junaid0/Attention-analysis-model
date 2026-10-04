"""Official L2CS-Net (Ahmednull/L2CS-Net): ResNet-50, 90 bins, 4 deg, [-180, 180]."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck

IDX = torch.arange(90).float()


class L2CSNet(nn.Module):
    """Gaze pitch/yaw from a 224×224 face crop. Same architecture as L2CS-Net."""

    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3), num_bins: int = 90) -> None:
        super().__init__()
        self.inplanes = 64
        self.num_bins = num_bins
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).flatten(1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)

    @torch.inference_mode()
    def predict_degrees(self, face_batch: torch.Tensor) -> torch.Tensor:
        """Return (N, 2) yaw, pitch in degrees.

        Matches Ahmednull/L2CS-Net Pipeline.predict_gaze unpacking of the Gaze360 checkpoint.
        """
        gaze_pitch, gaze_yaw = self.forward(face_batch)
        idx = IDX.to(face_batch.device)
        pitch = (torch.softmax(gaze_pitch, dim=1) * idx).sum(1) * 4.0 - 180.0
        yaw = (torch.softmax(gaze_yaw, dim=1) * idx).sum(1) * 4.0 - 180.0
        return torch.stack([yaw, pitch], dim=1)

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "L2CSNet":
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device=str(map_location))
        else:
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            state = ckpt
            if isinstance(ckpt, dict):
                for key in ("model", "state_dict", "model_state_dict"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        state = ckpt[key]
                        break
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(cleaned, strict=False)
        self.eval()
        return self
