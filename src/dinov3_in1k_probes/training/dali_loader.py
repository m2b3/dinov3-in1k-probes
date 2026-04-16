"""NVIDIA DALI GPU data loader for ImageNet-1K.

Linux + CUDA only. Imports are deferred so the module can be imported
(but not called) on macOS for type-checking.
"""

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass

from dinov3_in1k_probes.data import IMAGENET_MEAN, IMAGENET_STD

log = logging.getLogger(__name__)

# DALI normalizes uint8 [0, 255] directly, so scale the [0, 1] constants.
_MEAN_U8 = [m * 255 for m in IMAGENET_MEAN]
_STD_U8 = [s * 255 for s in IMAGENET_STD]


@dataclass
class DALILoader:
    _iterator: Iterator
    images_per_epoch: int
    batch_size: int

    def __iter__(self) -> Iterator:
        return self._iterator


def create_loader(
    *,
    data_root: str,
    batch_size: int,
    image_size: int,
    training: bool = False,
    return_filenames: bool = False,
    num_threads: int = 16,
    device_id: int = 0,
) -> DALILoader:
    """Build a DALI pipeline yielding normalized (B,3,H,W) CUDA tensors.

    DALI pads the last batch via wraparound — callers must slice.
    """
    import cupy as cp
    import nvidia.dali.fn as fn
    import torch
    from nvidia.dali import pipeline_def
    from nvidia.dali.pipeline import DataNode
    from nvidia.dali.types import DALIDataType, DALIImageType

    reader_name = "TrainReader" if training else "ValReader"

    log.info("DALI: %s %dpx BS=%d training=%s threads=%d", data_root, image_size, batch_size, training, num_threads)

    @pipeline_def()
    def pipe():
        jpegs, labels = fn.readers.file(
            file_root=data_root, random_shuffle=training,
            name=reader_name, pad_last_batch=True,
        )
        images = fn.decoders.image(jpegs, device="mixed", output_type=DALIImageType.RGB)

        if training:
            images = fn.random_resized_crop(images, size=(image_size, image_size))
            flip = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.resize(images, mode="not_smaller", size=(image_size, image_size))
            flip = 0

        assert isinstance(images, DataNode)
        images = fn.crop_mirror_normalize(
            images, mirror=flip, crop=(image_size, image_size),
            dtype=DALIDataType.FLOAT, output_layout="CHW",
            mean=_MEAN_U8, std=_STD_U8,
        )
        return (images, labels, jpegs) if return_filenames else (images, labels)

    p = pipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id)  # pyright: ignore[reportCallIssue] — @pipeline_def transforms the signature
    p.build()

    epoch_size = p.epoch_size()[reader_name]
    n_batches = math.ceil(epoch_size / batch_size)
    log.info("  epoch_size=%d  n_batches=%d", epoch_size, n_batches)

    def iterator() -> Iterator[dict]:
        for _ in range(n_batches):
            outputs = p.run()
            if return_filenames:
                img_dali, lbl_dali, jpeg_dali = outputs
                fnames = [jpeg_dali[i].source_info() for i in range(len(jpeg_dali))]
            else:
                img_dali, lbl_dali = outputs
                fnames = None

            images = torch.as_tensor(cp.asarray(img_dali.as_tensor()), device="cuda")
            labels = torch.as_tensor(cp.asarray(lbl_dali.as_tensor()), device="cuda")
            torch.cuda.synchronize()

            batch: dict = {"images": images, "labels": labels}
            if fnames is not None:
                batch["filenames"] = fnames
            yield batch

    return DALILoader(_iterator=iterator(), images_per_epoch=epoch_size, batch_size=batch_size)
