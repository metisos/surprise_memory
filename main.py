import asyncio
import sys
from datetime import datetime
from typing import List, Dict, Any, Deque
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

###############################################################################
#                        PERSISTENT (META) MEMORY
###############################################################################
class PersistentMemory(nn.Module):
    """
    Stores a few learnable tokens that represent meta-knowledge embedding.
    By default, we keep these trainable for a "training" phase. Then, at
    inference, we freeze them by calling 'freeze()'.
    """
    def __init__(self, num_tokens=2, dim=64):
        super().__init__()
        # Randomly initialize the persistent tokens
        self.persistent_tokens = nn.Parameter(torch.randn(num_tokens, dim))

    def freeze(self):
        """Freeze persistent memory for inference (no more grad updates)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Un-freeze persistent memory for training."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self) -> torch.Tensor:
        """
        Returns a [num_tokens, dim] tensor of persistent memory tokens.
        """
        return self.persistent_tokens


###############################################################################
#                     DEEP NEURAL MEMORY MODULE (multi-layer)
###############################################################################
class DeepNeuralMemory(nn.Module):
    """
    The "long-term memory" in Titans. An MLP that we can update at test (inference)
    time to store new key->value associations using gradient-based rules.
    """
    def __init__(self, dim_in=64, hidden_dim=128, num_layers=3):
        super().__init__()
        # MLP that maps key -> predicted_value
        layers = []
        in_dim = dim_in
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final layer maps back to dimension of 'value'
        layers.append(nn.Linear(in_dim, dim_in))
        self.net = nn.Sequential(*layers)

        # We'll define two linear projections to transform raw embeddings
        # into key and value.
        self.key_proj = nn.Linear(dim_in, dim_in, bias=False)
        self.value_proj = nn.Linear(dim_in, dim_in, bias=False)

    def forward(self, x: torch.Tensor, mode="pred") -> torch.Tensor:
        """
        If mode="kv", interpret x as raw embeddings -> (key, value).
        If mode="pred", interpret x as keys -> MLP -> predicted values.
        """
        if mode == "kv":
            k = self.key_proj(x)
            v = self.value_proj(x)
            return k, v
        else:
            return self.net(x)


###############################################################################
#                          SIMPLE ATTENTION BLOCK
###############################################################################
class SimpleAttentionBlock(nn.Module):
    """
    A multi-head self-attention layer to serve as the short-term memory.
    We treat the combined [persistent tokens + short-term memory + current user]
    as a single sequence, then do standard scaled dot-product attention.
    """
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj   = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)

        # Final projection after attention
        self.out_proj   = nn.Linear(dim, dim, bias=False)

        self.scale = (dim // num_heads) ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch=1, seq_len, dim]
        Returns: [batch=1, seq_len, dim]
        """
        B, N, D = x.shape
        q = self.query_proj(x)  # [1, N, D]
        k = self.key_proj(x)    # [1, N, D]
        v = self.value_proj(x)  # [1, N, D]

        # Reshape for multi-head: [1, N, D] -> [1, heads, N, d_head]
        D_head = D // self.num_heads
        q = q.reshape(B, N, self.num_heads, D_head).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, D_head).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, D_head).transpose(1, 2)

        # Dot product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, heads, N, d_head]

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


###############################################################################
#                     TITANS AGENT WITH SHORT-TERM ATTENTION
###############################################################################
class TitansAgent:
    def __init__(
        self,
        groq_client,
        short_term_capacity=10,
        dim_in=64,
        hidden_dim=128,
        num_layers=3,              # Memory depth
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-3,
        surprise_threshold=1.0,    # Magnitude of gradient norm for gating
        chunk_size=2,             # Trigger memory update after every 2 "surprising" messages
        num_heads=4
    ):
        """
        :param surprise_threshold: if gradient norm >= this threshold,
            consider the token surprising enough to store in chunk_buffer.
        """
        self.groq = groq_client
        self.short_term_capacity = short_term_capacity
        self.recent_memories: Deque[Dict[str, Any]] = deque(maxlen=short_term_capacity)

        # We'll accumulate user embeddings in chunk_buffer, then do memory updates
        self.chunk_buffer: List[torch.Tensor] = []
        self.chunk_size = chunk_size
        self.surprise_threshold = surprise_threshold

        # Our "long-term" memory model
        self.memory_model = DeepNeuralMemory(
            dim_in=dim_in,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Optimizer for test-time memory updates
        self.optimizer = optim.SGD(
            self.memory_model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Persistent memory (meta-knowledge). By default, un-frozen
        # so that we could train them if needed. We'll freeze in inference below.
        self.persistent_memory = PersistentMemory(num_tokens=2, dim=dim_in)

        # Short-term attention block
        self.attn_block = SimpleAttentionBlock(dim=dim_in, num_heads=num_heads)

        self.dim_in = dim_in

        # If we want to freeze the persistent memory at inference time, do it here:
        self.persistent_memory.freeze()
        # If you had a training phase for persistent memory, you would call
        # self.persistent_memory.unfreeze() before training, then freeze again afterwards.

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Convert a text string into a [dim_in] float vector.
        A simple ASCII-based embedding for demonstration.
        """
        ascii_vals = [ord(c) for c in text[:self.dim_in]]
        ascii_vals += [0] * (self.dim_in - len(ascii_vals))
        vec = torch.tensor(ascii_vals, dtype=torch.float32) / 128.0
        return vec

    def _calculate_surprise(self, user_emb: torch.Tensor) -> float:
        """
        Compute the gradient norm w.r.t. memory parameters, measuring "surprise."
        1) We'll do a small forward pass with 'user_emb' as if we want to store it.
        2) We'll measure MSE( M(keys), values ) and compute its gradient magnitude.
        3) This gradient magnitude is our "surprise" score.
        """
        # Temporarily enable grad for memory params
        # We'll do a forward/backward pass, then restore.
        # The user_emb shape is [dim], so unsqueeze(0) for batch dimension
        emb_2d = user_emb.unsqueeze(0)  # [1, dim]

        old_requires_grad = []
        for p in self.memory_model.parameters():
            old_requires_grad.append(p.requires_grad)
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad.zero_()

        # Forward pass
        k, v = self.memory_model(emb_2d, mode="kv")       # k,v: [1, dim]
        pred_v = self.memory_model(k, mode="pred")        # [1, dim]
        loss = F.mse_loss(pred_v, v)

        # Backward pass
        loss.backward(retain_graph=True)

        # Accumulate grad norm
        grad_norm_sq = 0.0
        for p in self.memory_model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm_sq ** 0.5

        # Restore
        for p, was_req_grad in zip(self.memory_model.parameters(), old_requires_grad):
            p.requires_grad_(was_req_grad)
            if p.grad is not None:
                p.grad.zero_()

        return grad_norm

    async def process_input(self, user_input: str) -> str:
        """
        1) Keep track of user inputs in short-term memory (deque).
        2) Form an attention context: [persistent tokens + short-term memory + current user].
        3) Run short-term attention -> final user embedding.
        4) Calculate surprise (gradient norm). If >= surprise_threshold, store in chunk_buffer.
        5) If chunk_buffer hits chunk_size, run memory update (Delta Rule).
        6) Call external LLM to produce response.
        """
        # 1) Insert user input
        self.recent_memories.append({
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # 2) Build attention context
        pm_tokens = self.persistent_memory().unsqueeze(0)  # shape [1, num_tokens, dim]

        # Prepare short-term embeddings
        st_embeds = []
        for mem in self.recent_memories:
            emb = self._embed_text(mem['content'])
            st_embeds.append(emb)
        if st_embeds:
            st_tensor = torch.stack(st_embeds, dim=0).unsqueeze(0)  # [1, st_len, dim]
        else:
            st_tensor = torch.empty(1, 0, self.dim_in)

        # Current user input embedding
        user_emb = self._embed_text(user_input).unsqueeze(0).unsqueeze(0)

        # Combine all into a single sequence
        attention_input = torch.cat([pm_tokens, st_tensor, user_emb], dim=1)
        # shape: [1, (persistent + st_len + 1), dim]

        # 3) Run short-term attention
        attended_output = self.attn_block(attention_input)
        attended_user = attended_output[:, -1, :]  # shape [1, dim], final user embedding

        # 4) Adaptive gating: measure "surprise" via gradient norm
        single_emb = attended_user.squeeze(0).detach()  # [dim_in]
        surprise_score = self._calculate_surprise(single_emb)
        memory_log = f"(Surprise={surprise_score:.2f} vs. threshold={self.surprise_threshold})"

        if surprise_score >= self.surprise_threshold:
            self.chunk_buffer.append(single_emb)

        # 5) If chunk_buffer is large enough, do a batch memory update
        if len(self.chunk_buffer) >= self.chunk_size:
            batch_tensor = torch.stack(self.chunk_buffer, dim=0)  # [chunk_size, dim]
            self.chunk_buffer.clear()

            # (k, v) = memory_model(batch_tensor, "kv"), pred_v = memory_model(k, "pred")
            k, v = self.memory_model(batch_tensor, mode="kv")
            pred_v = self.memory_model(k, mode="pred")
            loss = F.mse_loss(pred_v, v)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            memory_log += f" -> Memory update (loss={loss.item():.4f})"

        # 6) Prepare short-term summary + prompt
        short_term_summary = "Short-Term Memory:\n"
        for i, mem in enumerate(self.recent_memories):
            short_term_summary += f"  {i+1}) {mem['content']}\n"

        system_prompt = (
            "You are a Titans-based agent with:\n"
            "- Short-term memory (recent context) via attention\n"
            "- Deep neural memory updated at test time (Delta Rule)\n"
            "- Persistent memory tokens storing meta-knowledge\n"
            "Respond naturally and helpfully.\n"
        )
        user_prompt = (
            f"{short_term_summary}\n"
            f"{memory_log}\n"
            f"User just said: {user_input}\n\n"
            "Please produce a helpful response based on these memories."
        )

        # External LLM call
        if not self.groq:
            return "[Error: No Groq client configured.]"

        completion = self.groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=256,
            top_p=1.0
        )
        return completion.choices[0].message.content


###############################################################################
#                          MAIN INTERACTIVE CHAT
###############################################################################
async def main():
    """
    Simple console loop:
    - We only feed the user's typed input to the agent.
    - The agent's output is printed out, but never fed back in as user input.
    - Type 'quit' or 'exit' to stop.
    """
    try:
        import groq
        Groq = groq.Groq
        print("[Info] Using real Groq client")
    except ImportError:
        print("[Warning] groq not installed; using a mock client.")

        class MockCompletions:
            async def create(self, **kwargs):
                return {
                    "choices": [{"message": {"content": "[MOCK RESPONSE]"}}]
                }

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class Groq:
            def __init__(self, api_key=None):
                self.chat = MockChat()

    groq_client = Groq(api_key="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")

    # Example usage: vary memory depth easily
    agent = TitansAgent(
        groq_client=groq_client,
        short_term_capacity=10,
        dim_in=64,
        hidden_dim=128,
        num_layers=3,            # Memory depth
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-3,
        surprise_threshold=1.0,  # The bigger the threshold, the fewer updates
        chunk_size=2,
        num_heads=4
    )

    print("=== Welcome to the upgraded TitansAgent interactive chat ===")
    print("Type 'quit' or 'exit' to stop.\n")
    print("**Reminder**: Please do NOT paste the agent's output back in as user input.\n")

    while True:
        try:
            user_text = input("User: ").strip()
            if not user_text:
                continue
            if user_text.lower() in ("quit", "exit"):
                print("[System] Goodbye!")
                break

            response = await agent.process_input(user_text)
            print(f"[TitansAgent] {response}\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[System] Exiting...")
            break

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
