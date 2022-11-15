import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

class ClipLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale, i_labels=None, t_labels=None):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        i_labels = i_labels if i_labels is not None else torch.arange(num_logits, device=device, dtype=torch.long)
        t_labels = t_labels if t_labels is not None else torch.arange(num_logits, device=device, dtype=torch.long)

        total_loss = (F.cross_entropy(logits_per_image, i_labels) +
                      F.cross_entropy(logits_per_text, t_labels)
                      ) / 2
        return total_loss