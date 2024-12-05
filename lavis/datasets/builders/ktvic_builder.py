from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.ktvic_caption_datasets import (
    KTViCCapDataset,
    KTViCCapEvalDataset
)

from lavis.common.registry import registry


@registry.register_builder("ktvic_caption")
class KTViCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = KTViCCapDataset
    eval_dataset_cls = KTViCCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ktvic/defaults_cap.yaml"
    }