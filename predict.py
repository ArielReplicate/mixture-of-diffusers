import subprocess
import torch

subprocess.run(["cp", "-r", "huggingface", "/root/.cache"])

from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion, StableDiffusionTilingPipeline


from cog import BasePredictor, Path, Input, BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Predictor(BasePredictor):
    def setup(self):
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                         num_train_timesteps=1000)
        self.pipeline = StableDiffusionCanvasPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to("cuda:0")

    def predict(
        self,
        prompts: str = Input(description="Prompts separated by ';'",
                                       default="A charming house in the countryside, by jakub rozalski, "
                                               "sunset lighting, elegant, highly detailed, smooth, sharp focus, "
                                               "artstation, stunning masterpiece;"
                                               "A dirt road in the countryside crossing pastures, by jakub rozalski,"
                                               " sunset lighting, elegant, highly detailed, smooth, sharp focus, "
                                               "artstation, stunning masterpiece;"
                                               "An old and rusty giant robot lying on a dirt road, by jakub rozalski, "
                                               "dark sunset lighting, elegant, highly detailed, smooth, sharp focus,"
                                               " artstation, stunning masterpiece"),

        canvas_height: int = Input(description="Height of the output image", default=640),
        canvas_width: int = Input(description="Widthht of the output image", default=1408),
        y0_values: str = Input(description="Top row values separated by ';'", default='0;0;0'),
        x0_values: str = Input(description="Leftmost column values separated by ';'", default='0;384;768'),
        y1_values: str = Input(description="Bottom row values separated by ';'", default='640;640;640'),
        x1_values: str = Input(description="Rightmost column values separated by ';'", default='640;1024;1408'),
        seed: int = Input(description="Random seed", default=7178915308),
        num_inference_steps: int = Input(description="Backward diffusion steps", default=50)
    ) -> Path:

        prompts = str(prompts).split(';')
        y0_values = str(y0_values).split(';')
        x0_values = str(x0_values).split(';')
        y1_values = str(y1_values).split(';')
        x1_values = str(x1_values).split(';')
        assert len(prompts) == len(y0_values) == len(x0_values) == len(y1_values) == len(x1_values), "One of the inputs has a different number of ';' sepearted values"

        # Mixture of Diffusers generation
        regions = []
        for (y0, x0, y1, x1, prompt) in zip(y0_values, x0_values, y1_values, x1_values, prompts):
            # print(y0, x0, y1, x1, prompt)
            regions.append(Text2ImageRegion(int(y0), int(y1), int(x0), int(x1), guidance_scale=8, prompt=prompt))
        image = self.pipeline(
            canvas_height=int(canvas_height),
            canvas_width=int(canvas_width),
            regions=regions,
            num_inference_steps=int(num_inference_steps),
            seed=int(seed),
        )["sample"][0]


        out_path = "output.png"
        image.save(out_path)
        return Path(out_path)