import numpy as np
import torch



class DataCollator:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    def __init__(self, tokenizer, mlm_probability, vt_match_ratio,
                 num_frames=16, video_image_size=224, video_patch_size=16, tubelet_size=2, video_mask_ratio=0.8):
        self.tokenizer = tokenizer
        self.shuffle_ratio = 1 - vt_match_ratio
        self.mlm_probability = mlm_probability
        self.return_tensors = "pt"

        self.num_frames = num_frames
        self.video_image_size = video_image_size
        self.video_patch_size = video_patch_size
        self.tubelet_size = tubelet_size
        self.video_mask_ratio = video_mask_ratio

    def call(self, batch):

        batch, vt_match_labels, t_constrative_label, v_constrative_label = self.shuffle_match_video_text(batch)

        keys = list(batch[0].keys())
        length = len(batch)
        batch = {k: torch.stack([torch.tensor(i[k]) for i in batch], dim=0) for k in keys}
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"],
                                                                     special_tokens_mask=special_tokens_mask)

        num_patches_per_frame = (self.video_image_size // self.video_patch_size) ** 2
        seq_length = (self.num_frames // self.tubelet_size) * num_patches_per_frame
        bool_masked_pos = (torch.randn(1, seq_length) < self.video_mask_ratio).repeat(length, 1)
        batch["bool_masked_pos"] = bool_masked_pos

        batch['vt_match_labels'] = vt_match_labels.to(torch.long)
        batch['v_constrative_label'] = v_constrative_label.to(torch.long)
        batch['t_constrative_label'] = t_constrative_label.to(torch.long)

        return batch

    def shuffle_match_video_text(self, batch):

        bsz = len(batch)
        data_1, data_2 = [], []
        for b in batch:
            data_1.append({'video_pixel_values': b.get('video_pixel_values')})
            b.pop('video_pixel_values')
            data_2.append(b)

        shuffle_num = int(bsz * self.shuffle_ratio)
        data_1, data_2 = np.array(data_1), np.array(data_2)
        seg_1, seg_2 = np.arange(bsz), np.arange(bsz)

        shuf_seg_1, un_shuf_seg_1 = seg_1[:shuffle_num], seg_1[shuffle_num:]
        shuf_seg_2, un_shuf_seg_2 = seg_2[:shuffle_num], seg_2[shuffle_num:]

        un_shuf_labels = torch.ones(bsz - shuffle_num)
        np.random.shuffle(shuf_seg_1)
        np.random.shuffle(shuf_seg_2)

        shuf_labels = torch.tensor((shuf_seg_1 == shuf_seg_2).astype(np.int))
        vt_match_labels = torch.cat([shuf_labels, un_shuf_labels], dim=0)

        data_1 = np.hstack([data_1[shuf_seg_1], data_1[un_shuf_seg_1]])
        data_2 = np.hstack([data_2[shuf_seg_2], data_2[un_shuf_seg_2]])

        seg_1 = torch.tensor(np.hstack([shuf_seg_1, un_shuf_seg_1]))
        seg_2 = torch.tensor(np.hstack([shuf_seg_2, un_shuf_seg_2]))

        v_constrative_label = torch.tensor([torch.argwhere(seg_2 == i)[0][0] for i in seg_1])
        t_constrative_label = torch.tensor([torch.argwhere(seg_1 == i)[0][0] for i in seg_2])

        batch = []
        for i, j in zip(data_1, data_2):
            i.update(j)
            batch.append(i)

        return batch, vt_match_labels, v_constrative_label, t_constrative_label

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        while not torch.any(labels != -100):
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens with no pad word

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self, features):
        return self.call(features)