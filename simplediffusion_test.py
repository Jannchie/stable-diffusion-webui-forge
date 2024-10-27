from modules import shared
shared.options_templates = {}
from pathlib import Path
import unittest
import simplediffusion.modules.sd_models
from simplediffusion.processing import process_images
from simplediffusion.processing import StableDiffusionProcessingTxt2Img



class TestSimpleDiffusion(unittest.TestCase):
    def test_simple_diffusion(self):
        self.assertTrue(True)

    def test_txt_to_img(self):
        checkpoint_path = Path(
            R"E:\webui_forge_cu121_torch21\webui\models\Stable-diffusion\animagine-xl-3.1.safetensors"
        )
        checkpoint_info = simplediffusion.modules.sd_models.CheckpointInfo(
            str(checkpoint_path)
        )
        sd_model = simplediffusion.modules.sd_models.load_model(checkpoint_info)
        p = StableDiffusionProcessingTxt2Img()
        p.sd_model = sd_model
        p.steps = 20
        p.seed = 47
        p.prompt = "1girl, masterpiece, best quality, light brown background, from side, portrait, green eyes, tsurime, long hair, medium breasts, knees, headdress, string bikini, closed eyes"
        p.negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
        p.outpath_samples = "./outputs"
        p.width = 832
        p.cfg_scale = 5.0
        p.height = 1216
        res = process_images(p)
        res.images[0].save("outputs/result.png")
        p.prompt = "a cat in dark room"
        res = process_images(p)
        res.images[0].save("outputs/result2.png")


if __name__ == "__main__":
    unittest.main()
