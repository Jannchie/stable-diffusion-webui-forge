import logging
import time
from pathlib import Path

import cv2
import numpy as np
import safetensors
import torch

from modules.upscaler import UpscalerNone
from simplediffusion.processing import StableDiffusionProcessingTxt2Img


from . import checkpoint_pickle
from PIL import Image
from PIL.Image import Resampling


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C in [1, 3, 4]
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(resize_mode, im, width, height, upscaler_name=None, force_RGBA=False):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """
    upscaler_for_img2img = None
    upscaler_name = upscaler_name or upscaler_for_img2img
    sd_upscalers = [UpscalerNone]

    def resize(im, w, h):
        if (
            upscaler_name is None
            or upscaler_name == "None"
            or im.mode == "L"
            or force_RGBA
        ):
            return im.resize((w, h), resample=Resampling.LANCZOS)

        scale = max(w / im.width, h / im.height)
        if scale > 1.0:
            upscalers = [x for x in sd_upscalers if x.name == upscaler_name]
            if not upscalers:
                upscaler = sd_upscalers[0]
                print(
                    f"could not find upscaler named {upscaler_name or '<empty string>'}, using {upscaler.name} as a fallback"
                )
            else:
                upscaler = upscalers[0]

            im = upscaler.scaler.upscale(im, scale, upscaler.data_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=Resampling.LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGBA" if force_RGBA else "RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGBA" if force_RGBA else "RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(
                    resized.resize((fill_width, height), box=(0, 0, 0, height)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

    return res


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    img = input_image if skip_hwc3 else HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255.0 * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def align_dim_latent(x: int) -> int:
    """Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8


def calculate_image_dimensions(p):
    """Returns (h, w, hr_h, hr_w, has_high_res_fix)."""
    h = align_dim_latent(p.height)
    w = align_dim_latent(p.width)
    has_high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(
        p, "enable_hr", False
    )
    if has_high_res_fix:
        if p.hr_resize_x == 0 and p.hr_resize_y == 0:
            hr_y = int(p.height * p.hr_scale)
            hr_x = int(p.width * p.hr_scale)
        else:
            hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
        hr_y = align_dim_latent(hr_y)
        hr_x = align_dim_latent(hr_x)
    else:
        hr_y = h
        hr_x = w
    return h, w, hr_y, hr_x, has_high_res_fix


def load_torch_file(ckpt: Path | str, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if str(ckpt).lower().endswith(".safetensors"):
        return safetensors.torch.load_file(ckpt, device=device.type)
    if safe_load and "weights_only" not in torch.load.__code__.co_varnames:
        print(
            "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
        )
        safe_load = False
    pl_sd = (
        torch.load(ckpt, map_location=device, weights_only=True)
        if safe_load
        else torch.load(
            ckpt,
            map_location=device,
            pickle_module=checkpoint_pickle,
        )
    )
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    return pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd


class Timer:
    def __init__(self, name="Time taken"):
        self.start = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type or exc_value or traceback:
            logging.info(f"{self.name}: Exception occurred.")

        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"{self.name}: {self.interval:.2f} seconds.")
