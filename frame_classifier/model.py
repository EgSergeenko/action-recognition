import torch
import torchvision


class FrameClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(
            weights='IMAGENET1K_V1',
        )
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(
                in_features=1280,
                out_features=num_classes,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
