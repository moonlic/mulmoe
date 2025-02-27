from models.dynaminc_moe import DynameicSegMoEPipeline

pipeline = DynameicSegMoEPipeline(
    r"D:\code program\practice&fun\segmoe\segmoe_config_sd.yaml", device="cuda"
)
prompt = "Bedroom scene with a bookcase, blue comforter and window. A bedroom with a bookshelf full of books. This room has a bed with blue sheets and a large bookcase A bed and a mirror in a small room. A bed room with a neatly made bed a window and a book shelf"
negative_prompt = "nsfw, bad quality, worse quality"
img = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]
img.save("out_dynamic_SegMoe.png")

