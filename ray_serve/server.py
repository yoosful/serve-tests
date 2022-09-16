from ray import serve

import cv2
import torch
import numpy as np
from io import BytesIO
from starlette.requests import Request
from typing import Dict, List
from datetime import datetime


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2, "num_gpus": 1})
class D2Model:
    def __init__(self):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.model = model
        self.cfg = cfg

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=3)
    async def handle_batch(self, image_bytes_list: List) -> List[str]:
        now = datetime.now().isoformat(timespec='milliseconds')

        # bytes to images
        original_images = []
        for image_bytes in image_bytes_list:
            image_np = np.frombuffer(image_bytes, np.uint8)
            original_images.append(cv2.imdecode(image_np, cv2.IMREAD_COLOR))

        # images to inputs
        inputs = []
        for original_image in original_images:
            if self.cfg.INPUT.FORMAT == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})

        # inputs to outputs
        with torch.no_grad():
            outputs = self.model(inputs)

        instances_list = [output["instances"].to("cpu") for output in outputs]

        # for i,original_image in enumerate(original_images):
        #     v = Visualizer(original_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        #     out = v.draw_instance_predictions(instances_list[i])
        #     out.save(f"./output_{now}_{i}.jpg")


        return [
            {
                "pred_classes": instances.pred_classes.tolist(),
                "pred_boxes": instances.pred_boxes.tensor.tolist(),
            } for instances in instances_list
        ]


    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        return await self.handle_batch(image_payload_bytes)


d2_model = D2Model.bind()
