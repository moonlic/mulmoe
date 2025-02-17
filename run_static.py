from models.segmoe import SegMoEPipeline

pipeline = SegMoEPipeline(
    r"D:\code program\practice&fun\segmoe\segmoe_config_sd.yaml", device="cuda"
)
prompt = "A bright, shadowless moon emerging faintly under the night sky."
negative_prompt = "nsfw, bad quality, worse quality"
img = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]
img.save("out_static_SegMoe.png")
