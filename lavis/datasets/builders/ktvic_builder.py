from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.ktvic_caption_datasets import (
    KTViCCapEvalDataset,
    KTViCNoCapsEvalDataset
)

from lavis.common.registry import registry


@registry.register_builder("ktvic_caption")
class KTViCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = KTViCCapEvalDataset
    eval_dataset_cls = KTViCNoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ktvic/defaults_cap.yaml"
    }