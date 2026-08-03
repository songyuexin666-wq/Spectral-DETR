import unittest


try:
    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    from rfdetr.datasets.coco import ConvertCoco
    from rfdetr.datasets.coco_eval import CocoEvaluator
    from rfdetr.models.matcher import HungarianMatcher

    HAS_DETECTION_DEPS = True
except (ImportError, RuntimeError):
    HAS_DETECTION_DEPS = False


@unittest.skipUnless(HAS_DETECTION_DEPS, "PyTorch detection dependencies are unavailable")
class CategoryMappingTests(unittest.TestCase):
    def test_non_contiguous_category_ids_become_contiguous_labels(self):
        converter = ConvertCoco(category_id_to_label={1: 0, 3: 1, 7: 2})
        image = Image.new("RGB", (20, 20))
        annotations = [
            {"bbox": [1, 1, 4, 4], "area": 16, "category_id": category_id}
            for category_id in (1, 3, 7)
        ]

        _, target = converter(
            image, {"image_id": 11, "annotations": annotations}
        )

        self.assertEqual(target["labels"].tolist(), [0, 1, 2])

    def test_evaluator_restores_original_ids_and_filters_extra_class(self):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 11, "width": 20, "height": 20}],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "one"},
                {"id": 3, "name": "three"},
                {"id": 7, "name": "seven"},
            ],
        }
        coco.createIndex()
        evaluator = CocoEvaluator(coco, ("bbox",))
        predictions = {
            11: {
                "boxes": torch.tensor(
                    [[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4], [3, 3, 5, 5]],
                    dtype=torch.float32,
                ),
                "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
                "labels": torch.tensor([0, 1, 2, 3]),
            }
        }

        results = evaluator.prepare_for_coco_detection(predictions)

        self.assertEqual([item["category_id"] for item in results], [1, 3, 7])

    def test_matcher_does_not_shift_batch_without_class_zero(self):
        matcher = HungarianMatcher(cost_class=1, cost_bbox=0, cost_giou=0)
        outputs = {
            "pred_logits": torch.tensor(
                [[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]]]
            ),
            "pred_boxes": torch.zeros((1, 3, 4)),
        }
        targets = [{"labels": torch.tensor([1, 2]), "boxes": torch.zeros((2, 4))}]

        source_indices, target_indices = matcher(outputs, targets)[0]
        matched_by_target = {
            int(target): int(source)
            for source, target in zip(source_indices, target_indices)
        }

        self.assertEqual(matched_by_target, {0: 1, 1: 2})


if __name__ == "__main__":
    unittest.main()
