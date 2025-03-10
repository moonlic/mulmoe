import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
import torch.nn.functional as F
import torch.nn as nn
import gc, os
import tqdm
import yaml
from collections import OrderedDict
from math import ceil
from copy import deepcopy
import json
import safetensors
from typing import Any

from sparse_moe import SparseMoeBlock


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = (
                    OrderedDict()
                )  # child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


def getActivation(activation, name):
    def hook(model, inp, output):
        activation[name] = inp

    return hook


class SegMoEPipeline:
    def __init__(self, config_or_path, **kwargs) -> Any:
        """
        Instantiates the SegMoEPipeline. SegMoEPipeline implements the Segmind Mixture of Diffusion Experts, efficiently combining Stable Diffusion and Stable Diffusion Xl models.

        Usage:

        from segmoe import SegMoEPipeline
        pipeline = SegMoEPipeline(config_or_path, **kwargs)

        config_or_path: Path to Config or Directory containing SegMoE checkpoint or HF Card of SegMoE Checkpoint.

        Other Keyword Arguments:
        torch_dtype: Data Type to load the pipeline in. (Default: torch.float16)
        variant: Variant of the Model. (Default: fp16)
        device: Device to load the model on. (Default: cuda)
        Other args supported by diffusers.DiffusionPipeline are also supported.

        For more details visit https://github.com/segmind/segmoe.
        """
        self.torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        self.use_safetensors = kwargs.pop("use_safetensors", True)
        self.variant = kwargs.pop("variant", "fp16")
        self.device = kwargs.pop("device", "cuda")
        if os.path.isfile(config_or_path):
            self.load_from_scratch(config_or_path, **kwargs)
        else:
            if not os.path.isdir(config_or_path):
                cached_folder = DiffusionPipeline.download(config_or_path, **kwargs)
            else:
                cached_folder = config_or_path
            unet = self.create_empty(cached_folder)
            unet.load_state_dict(
                safetensors.torch.load_file(
                    f"{cached_folder}/unet/diffusion_pytorch_model.safetensors"
                )
            )
            if self.config.get("type", "sdxl") == "sdxl":
                self.base_cls = StableDiffusionXLPipeline
            elif self.config.get("type", "sdxl") == "sd":
                self.base_cls = StableDiffusionPipeline
            else:
                raise NotImplementedError(
                    "Base class not yet supported, type should be one of ['sd','sdxl]"
                )
            self.pipe = self.base_cls.from_pretrained(
                cached_folder,
                unet=unet,
                torch_dtype=self.torch_dtype,
                use_safetensors=self.use_safetensors,
                **kwargs,
            )
            self.pipe.to(self.device)
            self.pipe.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
                # memory_format指定了张良在内存中的存储方式，上一行的内存格式是主要用于图像数据，特别是CNN
            )

    def load_from_scratch(self, config: str, **kwargs) -> None:
        # Load Config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            # with是上下文管理器，保证文件操作后能正确关闭，即使读取异常也并不影响，r为只读模式
            # YAML是PyYAML库中的一个方法，用于解析YAML文件并将其转换为python数据结构，f代表要加载的YAML文件
            # yaml.safeLoader是一个安全加载器，避免潜在恶意代码
        self.config = config
        if self.config.get("num_experts", None):
            self.num_experts = self.config["num_experts"]
        else:
            if self.config.get("experts", None):
                self.num_experts = len(self.config["experts"])
            else:
                if self.config.get("loras", None):
                    self.num_experts = len(self.config["loras"])
                else:
                    self.num_experts = 1
        num_experts_per_tok = self.config.get("num_experts_per_tok", 1)
        # 从字典中获取值，如果不存在使用默认值1
        self.config["num_experts_per_tok"] = num_experts_per_tok
        moe_layers = self.config.get("moe_layers", "attn")
        self.config["moe_layers"] = moe_layers
        if self.config.get("type", "sdxl") == "sdxl":
            self.base_cls = StableDiffusionXLPipeline
            self.config["type"] = "sdxl"
        elif self.config.get("type", "sdxl") == "sd":
            self.base_cls = StableDiffusionPipeline
            self.config["type"] = "sd"
        else:
            raise NotImplementedError(
                "Base class not yet supported, type should be one of ['sd','sdxl]"
            )
        # Load Base Model
        if self.config["base_model"].startswith(
            "https://civitai.com/api/download/models/"
        ):
            os.makedirs("base", exist_ok=True)
            if not os.path.isfile("base/model.safetensors"):
                os.system(
                    "wget -O "
                    + "base/model.safetensors "
                    + self.config["base_model"]
                    + " --content-disposition"
                )
            self.config["base_model"] = "base/model.safetensors"
            self.pipe = self.base_cls.from_single_file(
                self.config["base_model"], torch_dtype=self.torch_dtype
            )
        elif os.path.isfile(self.config["base_model"]):
            self.pipe = self.base_cls.from_single_file(
                self.config["base_model"],
                torch_dtype=self.torch_dtype,
                use_safetensors=self.use_safetensors,
                **kwargs,
            )
        else:
            try:
                self.pipe = self.base_cls.from_pretrained(
                    self.config["base_model"],
                    torch_dtype=self.torch_dtype,
                    use_safetensors=self.use_safetensors,
                    variant=self.variant,
                    **kwargs,
                )
            except:
                self.pipe = self.base_cls.from_pretrained(
                    self.config["base_model"], torch_dtype=self.torch_dtype, **kwargs
                )
        # 定义上采样块和下采样块的起始、结束索引
        if self.base_cls == StableDiffusionPipeline:
            self.up_idx_start = 1
            self.up_idx_end = len(self.pipe.unet.up_blocks)
            self.down_idx_start = 0
            self.down_idx_end = len(self.pipe.unet.down_blocks) - 1
        elif self.base_cls == StableDiffusionXLPipeline:
            self.up_idx_start = 0
            self.up_idx_end = len(self.pipe.unet.up_blocks) - 1
            self.down_idx_start = 1
            self.down_idx_end = len(self.pipe.unet.down_blocks)
        self.config["up_idx_start"] = self.up_idx_start
        self.config["up_idx_end"] = self.up_idx_end
        self.config["down_idx_start"] = self.down_idx_start
        self.config["down_idx_end"] = self.down_idx_end

        # TODO: Add Support for Scheduler Selection
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)

        # Load Experts
        experts = []
        positive = []
        negative = []
        # 对加载的模型进行下载
        if self.config.get("experts", None):
            for i, exp in enumerate(self.config["experts"]):
                positive.append(exp["positive_prompt"])
                negative.append(exp["negative_prompt"])
                if exp["source_model"].startswith(
                    "https://civitai.com/api/download/models/"
                ):
                    try:
                        if not os.path.isfile(f"expert_{i}/model.safetensors"):
                            os.makedirs(f"expert_{i}", exist_ok=True)
                            if not os.path.isfile(f"expert_{i}/model.safetensors"):
                                os.system(
                                    f"wget {exp['source_model']} -O "
                                    + f"expert_{i}/model.safetensors "
                                    + " --content-disposition"
                                )
                        exp["source_model"] = f"expert_{i}/model.safetensors"
                        expert = self.base_cls.from_single_file(
                            exp["source_model"], **kwargs
                        ).to(self.device, self.torch_dtype)
                    except Exception as e:
                        print(f"Expert {i} {exp['source_model']} failed to load")
                        print("Error:", e)
                elif os.path.isfile(exp["source_model"]):
                    expert = self.base_cls.from_single_file(
                        exp["source_model"],
                        torch_dtype=self.torch_dtype,
                        use_safetensors=self.use_safetensors,
                        variant=self.variant,
                        **kwargs,
                    )
                    expert.scheduler = DDPMScheduler.from_config(
                        expert.scheduler.config
                    )
                else:
                    try:
                        expert = self.base_cls.from_pretrained(
                            exp["source_model"],
                            torch_dtype=self.torch_dtype,
                            use_safetensors=self.use_safetensors,
                            variant=self.variant,
                            **kwargs,
                        )

                        # TODO: Add Support for Scheduler Selection
                        expert.scheduler = DDPMScheduler.from_config(
                            expert.scheduler.config
                        )
                    except:
                        expert = self.base_cls.from_pretrained(
                            exp["source_model"], torch_dtype=self.torch_dtype, **kwargs
                        )
                        expert.scheduler = DDPMScheduler.from_config(
                            expert.scheduler.config
                        )
                # 对LoRA模块加载和处理
                if exp.get("loras", None):
                    for j, lora in enumerate(exp["loras"]):
                        if lora.get("positive_prompt", None):
                            positive[-1] += " " + lora["positive_prompt"]
                        if lora.get("negative_prompt", None):
                            negative[-1] += " " + lora["negative_prompt"]
                        if lora["source_model"].startswith(
                            "https://civitai.com/api/download/models/"
                        ):
                            try:
                                os.makedirs(f"expert_{i}/lora_{i}", exist_ok=True)
                                if not os.path.isfile(
                                    f"expert_{i}/lora_{i}/pytorch_lora_weights.safetensors"
                                ):
                                    os.system(
                                        f"wget {lora['source_model']} -O "
                                        + f"expert_{i}/lora_{j}/pytorch_lora_weights.safetensors"
                                        + " --content-disposition"
                                    )
                                lora["source_model"] = f"expert_{j}/lora_{j}"
                                expert.load_lora_weights(lora["source_model"])
                                if len(exp["loras"]) == 1:
                                    expert.fuse_lora()
                            except Exception as e:
                                print(
                                    f"Expert{i} LoRA {j} {lora['source_model']} failed to load"
                                )
                                print("Error:", e)
                        else:
                            expert.load_lora_weights(lora["source_model"])
                            if len(exp["loras"]) == 1:
                                expert.fuse_lora()
                experts.append(expert)
        else:
            experts = [deepcopy(self.pipe) for _ in range(self.num_experts)]
        if self.config.get("experts", None):
            if self.config.get("loras", None):
                for i, lora in enumerate(self.config["loras"]):
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            os.makedirs(f"lora_{i}", exist_ok=True)
                            if not os.path.isfile(
                                f"lora_{i}/pytorch_lora_weights.safetensors"
                            ):
                                os.system(
                                    f"wget {lora['source_model']} -O "
                                    + f"lora_{i}/pytorch_lora_weights.safetensors"
                                    + " --content-disposition"
                                )
                            lora["source_model"] = f"lora_{i}"
                            self.pipe.load_lora_weights(lora["source_model"])
                            if len(self.config["loras"]) == 1:
                                self.pipe.fuse_lora()
                        except Exception as e:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                            print("Error:", e)
                    else:
                        self.pipe.load_lora_weights(lora["source_model"])
                        if len(self.config["loras"]) == 1:
                            self.pipe.fuse_lora()
        else:
            if self.config.get("loras", None):
                j = []
                n_loras = len(self.config["loras"])
                i = 0
                positive = [""] * len(experts)
                negative = [""] * len(experts)
                while n_loras:
                    n = ceil(n_loras / len(experts))
                    j += [i] * n
                    n_loras -= n
                    i += 1
                for i, lora in enumerate(self.config["loras"]):
                    positive[j[i]] += lora["positive_prompt"] + " "
                    negative[j[i]] += lora["negative_prompt"] + " "
                    if lora["source_model"].startswith(
                        "https://civitai.com/api/download/models/"
                    ):
                        try:
                            os.makedirs(f"lora_{i}", exist_ok=True)
                            if not os.path.isfile(
                                f"lora_{i}/pytorch_lora_weights.safetensors"
                            ):
                                os.system(
                                    f"wget {lora['source_model']} -O "
                                    + f"lora_{i}/pytorch_lora_weights.safetensors"
                                    + " --content-disposition"
                                )
                            lora["source_model"] = f"lora_{i}"
                            experts[j[i]].load_lora_weights(lora["source_model"])
                            experts[j[i]].fuse_lora()
                        except:
                            print(f"LoRA {i} {lora['source_model']} failed to load")
                    else:
                        experts[j[i]].load_lora_weights(lora["source_model"])
                        experts[j[i]].fuse_lora()
        ###################################以上都是加载部分#################################################################
        ###################################下面是使用sparse block进行替换ff和attention#################################################################
        # Replace FF and Attention Layers with Sparse MoE Layers
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(self.pipe.unet.down_blocks[i].attentions)):
                for k in range(
                    len(self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                self.pipe.unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        # FF Layers
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .ff
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)
                    if not moe_layers == "ff":
                        ## Attns
                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": self.num_experts,
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_q
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_k
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_v
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_q
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                            "out_dim": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_k
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.down_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_v
                                )
                            )
                        self.pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)

        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(self.pipe.unet.up_blocks[i].attentions)):
                for k in range(
                    len(self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        # 如果设置为前馈层
                        config = {
                            "hidden_size": next(
                                self.pipe.unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        # FF Layers
                        # 创建稀疏专家前馈层 的专家列表
                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .ff
                                )
                            )
                        # 使用稀疏moebolck替代原始的前馈层【把上采样模块中的注意力层中的transformer块中的前馈网络换成SMoeBlock】
                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].ff = SparseMoeBlock(config, layers)
                        # 关键代码点 把.ff层换成spareseMoeBlock

                    if not moe_layers == "ff":
                        # Attns
                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []
                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_q
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_k
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn1.to_v
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_q
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "out_dim": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }

                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_k
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": self.pipe.unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": len(experts),
                        }
                        layers = []

                        for l in range(len(experts)):
                            layers.append(
                                deepcopy(
                                    experts[l]
                                    .unet.up_blocks[i]
                                    .attentions[j]
                                    .transformer_blocks[k]
                                    .attn2.to_v
                                )
                            )

                        self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)

        # Routing Weight Initialization
        ####初始化MoE路由权重，对下采样的模块中注意力层、前馈层进行初始化处理
        if self.config.get("init", "hidden") == "hidden":
            gate_params = self.get_gate_params(experts, positive, negative)
            for i in range(self.down_idx_start, self.down_idx_end):
                for j in range(len(self.pipe.unet.down_blocks[i].attentions)):
                    for k in range(
                        len(
                            self.pipe.unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks
                        )
                    ):
                        # FF Layers
                        if not moe_layers == "attn":
                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[k].ff.gate.weight = nn.Parameter(
                                gate_params[f"d{i}a{j}t{k}"]
                            )

                        # Attns
                        if not moe_layers == "ff":
                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_q.gate.weight = nn.Parameter(
                                gate_params[f"sattnqd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_k.gate.weight = nn.Parameter(
                                gate_params[f"sattnkd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_v.gate.weight = nn.Parameter(
                                gate_params[f"sattnvd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_q.gate.weight = nn.Parameter(
                                gate_params[f"cattnqd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_k.gate.weight = nn.Parameter(
                                gate_params[f"cattnkd{i}a{j}t{k}"]
                            )

                            self.pipe.unet.down_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_v.gate.weight = nn.Parameter(
                                gate_params[f"cattnvd{i}a{j}t{k}"]
                            )

            for i in range(self.up_idx_start, self.up_idx_end):
                for j in range(len(self.pipe.unet.up_blocks[i].attentions)):
                    for k in range(
                        len(
                            self.pipe.unet.up_blocks[i].attentions[j].transformer_blocks
                        )
                    ):
                        # FF Layers
                        if not moe_layers == "attn":
                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[k].ff.gate.weight = nn.Parameter(
                                gate_params[f"u{i}a{j}t{k}"]
                            )
                        ###如果不是ff，则初始化，将attention层中的to_q,k,v中的gate.weight变为gate_params中的对应参数
                        if not moe_layers == "ff":
                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_q.gate.weight = nn.Parameter(
                                gate_params[f"sattnqu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_k.gate.weight = nn.Parameter(
                                gate_params[f"sattnku{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn1.to_v.gate.weight = nn.Parameter(
                                gate_params[f"sattnvu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_q.gate.weight = nn.Parameter(
                                gate_params[f"cattnqu{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_k.gate.weight = nn.Parameter(
                                gate_params[f"cattnku{i}a{j}t{k}"]
                            )

                            self.pipe.unet.up_blocks[i].attentions[
                                j
                            ].transformer_blocks[
                                k
                            ].attn2.to_v.gate.weight = nn.Parameter(
                                gate_params[f"cattnvu{i}a{j}t{k}"]
                            )
        ####清理内存
        self.config["num_experts"] = len(experts)
        remove_all_forward_hooks(self.pipe.unet)
        try:
            del experts
            del expert
        except:
            pass
        # Move Model to Device
        self.pipe.to(self.device)
        self.pipe.unet.to(
            device=self.device,
            dtype=self.torch_dtype,
            memory_format=torch.channels_last,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ###提供了可以使得外界声明实例并使用的调用方法
        """
        Inference the SegMoEPipeline.

        Calls diffusers.DiffusionPipeline forward with the keyword arguments. See https://github.com/segmind/segmoe#usage for detailed usage.
        """
        return self.pipe(*args, **kwds)

    def create_empty(self, path):
        with open(f"{path}/unet/config.json") as f:
            config = json.load(f)
        self.config = config["segmoe_config"]
        unet = UNet2DConditionModel.from_config(config)
        num_experts_per_tok = self.config["num_experts_per_tok"]
        num_experts = self.config["num_experts"]
        moe_layers = self.config["moe_layers"]
        self.up_idx_start = self.config["up_idx_start"]
        self.up_idx_end = self.config["up_idx_end"]
        self.down_idx_start = self.config["down_idx_start"]
        self.down_idx_end = self.config["down_idx_end"]
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(unet.down_blocks[i].attentions)):
                for k in range(
                    len(unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                unet.down_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        # FF Layers
                        #####重点
                        layers = [
                            unet.down_blocks[i].attentions[j].transformer_blocks[k].ff
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[k].ff = (
                            SparseMoeBlock(config, layers)
                        )
                    if not moe_layers == "ff":
                        ## Attns
                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                            "out_dim": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.down_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v
                        ] * num_experts
                        unet.down_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)
        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(unet.up_blocks[i].attentions)):
                for k in range(len(unet.up_blocks[i].attentions[j].transformer_blocks)):
                    if not moe_layers == "attn":
                        config = {
                            "hidden_size": next(
                                unet.up_blocks[i]
                                .attentions[j]
                                .transformer_blocks[k]
                                .ff.parameters()
                            ).size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        # FF Layers
                        layers = [
                            unet.up_blocks[i].attentions[j].transformer_blocks[k].ff
                        ] * num_experts
                        unet.up_blocks[i].attentions[j].transformer_blocks[k].ff = (
                            SparseMoeBlock(config, layers)
                        )

                    if not moe_layers == "ff":
                        # Attns
                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_q
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_k
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn1.to_v
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn1.to_v = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q.weight.size()[-1],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_q
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_q = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[-1],
                            "out_dim": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }

                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_k
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_k = SparseMoeBlock(config, layers)

                        config = {
                            "hidden_size": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[-1],
                            "out_dim": unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v.weight.size()[0],
                            "num_experts_per_tok": num_experts_per_tok,
                            "num_local_experts": num_experts,
                        }
                        layers = [
                            unet.up_blocks[i]
                            .attentions[j]
                            .transformer_blocks[k]
                            .attn2.to_v
                        ] * num_experts

                        unet.up_blocks[i].attentions[j].transformer_blocks[
                            k
                        ].attn2.to_v = SparseMoeBlock(config, layers)
        return unet

    def save_pretrained(self, path):
        """
        Save SegMoEPipeline to Disk.

        Usage:
        pipeline.save_pretrained(path)

        Parameters:
        path: Path to Directory to save the model in.
        """
        for param in self.pipe.unet.parameters():
            param.data = param.data.contiguous()
        self.pipe.unet.config["segmoe_config"] = self.config
        self.pipe.save_pretrained(path)
        safetensors.torch.save_file(
            self.pipe.unet.state_dict(),
            f"{path}/unet/diffusion_pytorch_model.safetensors",
        )

    def cast_hook(self, pipe, dicts):
        for i in range(self.down_idx_start, self.down_idx_end):
            for j in range(len(pipe.unet.down_blocks[i].attentions)):
                for k in range(
                    len(pipe.unet.down_blocks[i].attentions[j].transformer_blocks)
                ):
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff.register_forward_hook(getActivation(dicts, f"d{i}a{j}t{k}"))

                    ## Down Self Attns
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q.register_forward_hook(
                        getActivation(dicts, f"sattnqd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k.register_forward_hook(
                        getActivation(dicts, f"sattnkd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v.register_forward_hook(
                        getActivation(dicts, f"sattnvd{i}a{j}t{k}")
                    )

                    ## Down Cross Attns

                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q.register_forward_hook(
                        getActivation(dicts, f"cattnqd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k.register_forward_hook(
                        getActivation(dicts, f"cattnkd{i}a{j}t{k}")
                    )
                    pipe.unet.down_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v.register_forward_hook(
                        getActivation(dicts, f"cattnvd{i}a{j}t{k}")
                    )

        for i in range(self.up_idx_start, self.up_idx_end):
            for j in range(len(pipe.unet.up_blocks[i].attentions)):
                for k in range(
                    len(pipe.unet.up_blocks[i].attentions[j].transformer_blocks)
                ):
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].ff.register_forward_hook(getActivation(dicts, f"u{i}a{j}t{k}"))
                    ## Up Self Attns
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_q.register_forward_hook(
                        getActivation(dicts, f"sattnqu{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_k.register_forward_hook(
                        getActivation(dicts, f"sattnku{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn1.to_v.register_forward_hook(
                        getActivation(dicts, f"sattnvu{i}a{j}t{k}")
                    )

                    ## Up Cross Attns
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_q.register_forward_hook(
                        getActivation(dicts, f"cattnqu{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_k.register_forward_hook(
                        getActivation(dicts, f"cattnku{i}a{j}t{k}")
                    )
                    pipe.unet.up_blocks[i].attentions[j].transformer_blocks[
                        k
                    ].attn2.to_v.register_forward_hook(
                        getActivation(dicts, f"cattnvu{i}a{j}t{k}")
                    )

    @torch.no_grad
    def get_hidden_states(self, model, positive, negative, average: bool = True):
        intermediate = {}
        self.cast_hook(model, intermediate)
        with torch.no_grad():
            _ = model(positive, negative_prompt=negative, num_inference_steps=25)
        hidden = {}
        for key in intermediate:
            hidden_states = intermediate[key][0][-1]
            if average:
                # use average over sequence
                hidden_states = hidden_states.sum(dim=0) / hidden_states.shape[0]
            else:
                # take last value
                hidden_states = hidden_states[:-1]
            hidden[key] = hidden_states.to(self.device)
        del intermediate
        gc.collect()
        torch.cuda.empty_cache()
        return hidden

    @torch.no_grad
    def get_gate_params(
        self,
        experts,
        positive,
        negative,
    ):
        gate_vects = {}
        for i, expert in enumerate(tqdm.tqdm(experts, desc="Expert Prompts")):
            expert.to(self.device)
            expert.unet.to(
                device=self.device,
                dtype=self.torch_dtype,
                memory_format=torch.channels_last,
            )
            hidden_states = self.get_hidden_states(expert, positive[i], negative[i])
            del expert
            gc.collect()
            torch.cuda.empty_cache()
            for h in hidden_states:
                if i == 0:
                    gate_vects[h] = []
                hidden_states[h] /= (
                    hidden_states[h].norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                )
                gate_vects[h].append(hidden_states[h])
        for h in hidden_states:
            gate_vects[h] = torch.stack(
                gate_vects[h], dim=0
            )  # (num_expert, num_layer, hidden_size)
            gate_vects[h].permute(1, 0)

        return gate_vects
