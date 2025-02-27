import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


class SparseMoeBlock(nn.Module):
    def __init__(self, config, experts):
        super().__init__()
        self.hidden_dim = config["hidden_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.out_dim = config.get("out_dim", self.hidden_dim)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([deepcopy(exp) for exp in experts])

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, sequence_length, f_map_sz = hidden_states.shape
        hidden_states = hidden_states.view(-1, f_map_sz)
        # 将输入展平以便与门控层匹配
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        # 由gate函数生成每个输入token对应的专家分数，然后选择分数最高的top_k
        ##此处的分数理解为整体分数
        _, selected_experts = torch.topk(
            router_logits.sum(dim=0, keepdim=True), self.top_k, dim=1
        )
        # 使用softmax计算专家的路由权重(还是根据前面的分数)
        routing_weights = F.softmax(
            router_logits[:, selected_experts[0]], dim=1, dtype=torch.float
        )
        # we cast back to the input dtype
        # 将张量routing_weights转换回[因为此时精度更高float32]与输入张量hidden_states相同类型
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Loop over all available experts in the model and perform the computation on each expert
        # 循环遍历选定的专家
        for i, expert_idx in enumerate(selected_experts[0].tolist()):
            expert_layer = self.experts[expert_idx]
            # 计算每个专家的加权输出
            current_hidden_states = routing_weights[:, i].view(
                batch_size * sequence_length, -1
            ) * expert_layer(hidden_states)

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # 累计各个专家的加权输出
            # 结果融合-->states累计误差
            final_hidden_states = final_hidden_states + current_hidden_states
        # 调整输出形状
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, self.out_dim
        )
        return final_hidden_states


class DynamicSparseMoeBlock(SparseMoeBlock):
    # 继承，如果不是重新定义则基本全部继承，初始化也是如此，会沿袭父类的初始化方法，因为使用了superinit
    def __init__(self, config, experts):
        super().__init__(config, experts)
        # 增强门控网络,尝试使用非线性的方式
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, 128), nn.GELU(), nn.Linear(128, self.num_experts)
        )

    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.shape
        x = hidden_states.view(-1, dim)  # (B*S, D)

        # Token级路由决策
        router_logits = self.gate(x)  # (B*S, E)
        scores, experts_idx = torch.topk(router_logits, self.top_k, dim=-1)  # (B*S, K)
        weights = torch.softmax(scores, dim=-1)  # (B*S, K)

        # 创建专家掩码矩阵
        expert_mask = torch.zeros(
            x.size(0), self.num_experts, device=x.device, dtype=x.dtype
        )  # (B*S, E)
        expert_mask.scatter_(1, experts_idx, weights)  # 散射填充

        # 批量计算所有专家
        expert_outputs = torch.stack(
            [exp(x) for exp in self.experts], dim=1
        )  # (B*S, E, D)

        # 加权聚合
        output = (expert_mask.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B*S, D)

        return output.view(batch_size, seq_len, -1)
