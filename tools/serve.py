import asyncio
import io
import logging
from asyncio import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import torch
from PIL import Image
from aiohttp import web
from detectron2.engine import default_argument_parser, DefaultPredictor
from pandas import np

from tools.train_net import setup

routes = web.RouteTableDef()
thread_pool = ThreadPoolExecutor()
inference_queue = asyncio.Queue()


class BatchPredictor(DefaultPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def __call__(self, original_images: List):
        """
        Args:
            original_image (List[np.ndarray]): images of shape (H, W, C) (in BGR order).
        Returns:
            predictions (List[dict]): the output of the model
        """
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_images = [x[:, :, ::-1] for x in original_images]
        height, width = original_images[0].shape[:2]
        images = [self.transform_gen.get_transform(x).apply_image(x) for x in original_images]
        images = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1)) for x in images]

        inputs = [{"image": x, "height": height, "width": width} for x in images]
        predictions = self.model(inputs)
        return predictions


def _collect_future_results(results: List[Future]):
    predictions = [x.result() for x in results]

    ret = []
    for pred in predictions:
        instances = pred["instances"]
        ret.append({
            "filename": pred["filename"],
            "scores": instances.scores.tolist(),
            "classes": instances.pred_classes.tolist(),
            "boxes": instances.pred_boxes.tensor.tolist(),
        })
    return ret


def _load_image(b):
    try:
        img = Image.open(io.BytesIO(b))
        img = img.convert("RGB")
        return np.asarray(img)
    except:
        return None


@routes.post("/inference")
async def inference(request: web.Request):
    multipart = await request.multipart()
    loop = asyncio.get_event_loop()
    results = []
    while True:
        file = await multipart.next()
        if file is None:
            break

        b = await file.read()
        fn = file.filename

        image = await loop.run_in_executor(thread_pool, lambda: _load_image(b))

        if image is None:
            logging.warning("Failed to open an image. Skip.")
            continue

        result = loop.create_future()
        results.append(result)
        await inference_queue.put((image, fn, result))

    logging.info(f"Inference request contains: {len(results)} images to be processed")

    if len(results) == 0:
        return web.json_response([])

    await asyncio.wait(results, return_when=asyncio.ALL_COMPLETED)

    return web.json_response(_collect_future_results(results))


async def _queue_get(q: asyncio.Queue, at_most: int, at_least: int = 1):
    ret = []
    for i in range(at_least):
        ret.append(await q.get())

    for i in range(at_most - at_least):
        try:
            ret.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return ret


async def coro_detector(cfg):
    predictor = BatchPredictor(cfg)
    loop = asyncio.get_event_loop()
    while True:
        inputs = await _queue_get(inference_queue, cfg.SOLVER.IMS_PER_BATCH)

        model_input = [x[0] for x in inputs]
        predictions = await loop.run_in_executor(thread_pool, lambda: predictor(model_input))
        for (image, filename, result), prediction in zip(inputs, predictions):
            result: Future
            result.set_result({"filename": filename, "instances": prediction["instances"].to("cpu")})


def serve(cfg, host="127.0.0.1", port=10250):
    app = web.Application()
    app.add_routes(routes)
    asyncio.get_event_loop().create_task(coro_detector(cfg))
    web.run_app(app, host=host, port=port)


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=10250, type=int)
    args = parser.parse_args()

    cfg = setup(args)
    serve(cfg, args.host, args.port)
