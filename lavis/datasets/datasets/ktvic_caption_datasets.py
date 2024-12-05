import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset

KTViCCapDataset = CaptionDataset
KTViCCapInstructDataset = CaptionInstructDataset

class KTViCCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print(f'In KTViCCapDataset: 1st annotation keys: {self.annotation[0].keys()}')

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        # image_path = os.path.join(self.vis_root, f"{ann['image_id']:011}.jpg")
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        # img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
        img_id = ann["image_id"]

        return {
            "image": image,
            "image_id": img_id,
            "text_input": "một bức ảnh về ",
            "text_output": ann["caption"],
            "instance_id": ann["instance_id"],
        }
        # return {
        #     "image": image,
        #     "image_id": img_id,
        #     "caption": caption,
        # }


class KTViCCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print(f'In KTViCNoCapsEvalDataset: 1st annotation keys: {self.annotation[0].keys()}')

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print(f'ann keys: {ann.keys()}')
        # print(f'ann["sample_id"]: {ann["sample_id"]}')
        # print(f'ann["data"]: {ann["data"]}')
        # print(f'ann["instance_id"]: {ann["instance_id"]}')
        
        image_path = os.path.join(self.vis_root, ann["image"])
        # image_path = os.path.join(self.vis_root, f"{ann['image_id']:011}.jpg")
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        # captions = self.search_captions(img_id)

        return {
            "image": image,
            "text_input": "một bức ảnh về ",
            "text_output": ann["caption"],
            "instance_id": ann["instance_id"],
        }

    # def search_captions(self, img_id):
    #     """
    #     @tcm: In KTVIC, there are 5 captions per image. In addition, self.annotation is an array of dictionaries
    #     of each image-caption pair with increase image_id. Thus, we can binary search on this list to find captions of
    #     the image with img_id.
    #     """
    #     left = 0
    #     right = len(self.annotation) - 1
    #     while left != right:
    #         mid = (left + right) // 2
    #         if self.annotation[mid]["image_id"] < img_id:
    #             left = mid + 1
    #         else:
    #             right = mid
        
    #     # we need to make sure that left-th annotation is the first annotation of the image with img_id and the
    #     # next 4 captions are of this image too
    #     assert self.annotation[left]["image_id"] == img_id and \
    #             left+4 < len(self.annotation) and \
    #             self.annotation[left+4]["image_id"] == img_id

    #     return self.annotation[left:left+5]