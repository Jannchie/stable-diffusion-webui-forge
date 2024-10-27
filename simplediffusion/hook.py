import torch

from modules.processing import StableDiffusionProcessing


class Hook:

    def apply_on_before_component_callback(self):
        pass

    def before_process(self, p: StableDiffusionProcessing):
        pass

    def process(self, p: StableDiffusionProcessing):
        pass

    def before_process_batch(self, *args, **kwargs):
        pass

    def after_extra_networks_activate(self, *args, **kwargs):
        pass

    def process_batch(self, *args, **kwargs):
        pass

    def process_before_every_sampling(
        self,
        process: StableDiffusionProcessing,
        x: torch.Tensor,
        noise: torch.Tensor,
        conditioning: torch.Tensor,
        unconditional_conditioning: torch.Tensor,
    ):
        pass

    def postprocess(self, *args, **kwargs):
        pass

    def postprocess_batch(self, *args, **kwargs):
        pass

    def postprocess_batch_list(self, *args, **kwargs):
        pass

    def post_sample(self, *args, **kwargs):
        pass

    def on_mask_blend(self, *args, **kwargs):
        pass

    def postprocess_image(self, *args, **kwargs):
        pass

    def postprocess_maskoverlay(self, *args, **kwargs):
        pass

    def postprocess_image_after_composite(self, *args, **kwargs):
        pass

    def before_component(self, *args, **kwargs):
        pass

    def after_component(self, *args, **kwargs):
        pass
