

import numpy as np
import torch
from transformers import BatchFeature
from transformers.utils import is_torch_tensor


class AudioFeatureExtractor:
    def __init__(self, do_normalize=True, waveform_mean=0.5, waveform_std=0.5):
        self.do_normalize = do_normalize
        self.waveform_mean = waveform_mean
        self.waveform_std = waveform_std

    def normalize(self, waveform, mean, std):
        return ((waveform - mean) / std)

    def __call__(self, waveforms, return_tensors=None):
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(waveforms, np.ndarray) or is_torch_tensor(waveforms):
            valid_images = True
        elif isinstance(waveforms, (list, tuple)):
            if len(waveforms) == 0 or isinstance(waveforms[0], np.ndarray) or is_torch_tensor(waveforms[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(waveforms, (list, tuple))
            and (isinstance(waveforms[0], np.ndarray) or is_torch_tensor(waveforms[0]))
        )

        if not is_batched:
            waveforms = [waveforms]

        # transformations (normalization)
        if self.do_normalize:
            waveforms = [self.normalize(waveform=waveform, mean=self.waveform_mean, std=self.waveform_std) for waveform in waveforms]

        # return as BatchFeature
        data = {"pixel_values": waveforms}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        encoded_inputs['pixel_values'] = encoded_inputs['pixel_values'].to(torch.float32)
        return encoded_inputs
