import torch.nn.functional as F

from ultralytics.yolo.utils import ops, LOGGER
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class YOLOEmbeddingsPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        embedding = preds[1]
        embedding = F.adaptive_avg_pool2d(embedding, 2).flatten(1)
        return embedding

    @smart_inference_mode()
    def embed(self, source=None, model=None, verbose=True):
        """Streams real-time inference on camera feed and saves results to file."""
        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(
                imgsz=(
                    1 if self.model.pt or self.model.triton else self.dataset.bs,
                    3,
                    *self.imgsz,
                )
            )
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = (
            0,
            [],
            None,
            (ops.Profile(), ops.Profile(), ops.Profile()),
        )
        for batch in self.dataset:
            path, im0s, _, _ = batch
            if verbose:
                LOGGER.info(path[0])
            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, embed_from=-1)

            with profilers[2]:
                embeddings = self.postprocess(preds, im, im0s)

            return embeddings
            # yielding seems pointless as this is designed specifically to be used in for loops,
            # batching with embed_func would make things complex
