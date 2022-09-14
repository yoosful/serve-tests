from ray import serve

import cv2
import numpy as np
from io import BytesIO
from starlette.requests import Request
from typing import Dict

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.2})
class D2Model:
    def __init__(self):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg = cfg
        self.model = DefaultPredictor(cfg)

    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        image_np = np.frombuffer(image_payload_bytes, np.uint8)
        cv2_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # print("[1/2] Parsed image data!")

        outputs = self.model(cv2_image)
        # print("[2/2] Inference done!")

        instances = outputs["instances"].to("cpu")

        v = Visualizer(cv2_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(instances)
        out.save("./output.jpg")

        return {
            "pred_classes": instances.pred_classes.tolist(),
            "pred_boxes": instances.pred_boxes.tensor.tolist(),
        }


d2_model = D2Model.bind()
