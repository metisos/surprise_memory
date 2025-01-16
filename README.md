# **Titans Agent with Adaptive Test-Time Memory**

**Author**: *Christian Johnson*  
**Date**: *01/16/2025*  

## **1. Overview**

This repository implements a prototype **Titans-based AI agent** that combines three crucial memory concepts:

1. **Persistent (Meta) Memory**  
2. **Short-Term (Attention) Memory**  
3. **Long-Term (Deep Neural) Memory**

The core idea is to allow an agent to *continuously update* its internal weights **at inference/test time** in response to the user’s inputs. This approach borrows from the research on “Titans” (short for “Test-time Inference with Training at test time Architecture for Neural memory System”), offering:

- A **deep neural memory** module for storing key → value associations (updated online).  
- A **short-term attention** module for immediate context.  
- A **persistent memory** store that remains invariant at inference (or can be optionally trained beforehand).

---

## **2. Script Structure**

### 1. **`PersistentMemory`**
- Stores a small set of learnable tokens representing *global meta-knowledge*.  
- By default, the tokens are frozen during inference (`freeze()` method).  
- If training is desired, unfreeze them (`unfreeze()`) before training.

### 2. **`DeepNeuralMemory`**
- A multi-layer perceptron (MLP) to learn *key → value* mappings at test time.  
- Each forward pass can operate in two modes:  
  - `mode="kv"`: Interpret input as raw embeddings to produce `(k, v)`.  
  - `mode="pred"`: Interpret input as keys, returning predicted values.  
- This module uses standard PyTorch linear layers and ReLUs.

### 3. **`SimpleAttentionBlock`**
- A multi-head self-attention mechanism, providing *short-term memory*.  
- Splits queries, keys, and values across multiple heads, computes softmax attention, then recombines them.  
- Used to focus on the most recent context (including persistent tokens + user’s short-term memory buffer).

### 4. **`TitansAgent`**
- Brings everything together. Key components:
  - **Short-term capacity**: Deque for recent user messages.  
  - **Long-term memory** (`DeepNeuralMemory`) updated online.  
  - **Persistent memory** (`PersistentMemory`) holding meta-knowledge tokens.  
  - **Attention block** for short-term context.  
  - **Test-time training** logic for deciding when and how to update the neural memory.  
- **Adaptive gating / “surprise”**:  
  - Computes a “surprise score” (gradient magnitude) for new user embeddings.  
  - If above `surprise_threshold`, the embedding is queued in a *chunk_buffer*.  
  - Whenever this buffer reaches `chunk_size`, we apply a *Delta Rule* update:  
    1. Convert embeddings → (keys, values).  
    2. Compare predicted values with actual values.  
    3. Take an SGD step to optimize the memory weights.

### 5. **`main()`**
- Runs a simple console-based chat loop.  
- Prompts for user input, processes it, and displays the agent’s response.  
- Type `quit` or `exit` to terminate.

---

## **3. Key Features**

1. **Adaptive Gating (“Surprise” Metric)**
   - The script measures gradient norm on a small forward/backward pass to decide if an input is “surprising.”  
   - Surprising inputs get stored in the *chunk_buffer* and eventually update the memory.

2. **Chunk-Based Updates**
   - Instead of updating memory on *every* token, the agent waits until `chunk_buffer` hits a certain size (`chunk_size`) to do a more efficient batched update.

3. **Separation of Memory Components**
   - **Persistent memory** for domain knowledge or meta tokens.  
   - **Short-term memory** (the attention-based mechanism) focusing on the immediate past.  
   - **Long-term memory** (the MLP) that actually changes weights at inference time.

4. **Extensibility**
   - Easily modify memory depth (`num_layers`) or gating threshold to adapt to different tasks.  
   - Swap in different front-end embedding logic (currently ASCII-based).  
   - Replace the external LLM call (`groq_client`) with any language model API.

---

## **4. Installation & Dependencies**

### 1. **Python Version**
- Python 3.7+ recommended.

### 2. **Dependencies**
- `torch` (PyTorch) for automatic differentiation, tensor operations, and optimizers.  
- `groq` (or a *mock* client if not installed).  
- `asyncio` (standard Python library for asynchronous I/O).

### 3. **Installation**
```bash
pip install torch
# If you have groq:
pip install groq
# or omit if you plan to rely on the mock client
5. Running the Script
1. Command
bash
Copy
Edit
python surprise.py
Where surprise.py is your main script file.

2. Interactive Chat
Once it starts, you’ll see a message:
arduino
Copy
Edit
=== Welcome to the upgraded TitansAgent interactive chat ===
Type 'quit' or 'exit' to stop.
Type your input at User:, and press Enter. The agent responds with [TitansAgent] ....
3. Terminating
Type quit or exit at the user prompt.
6. How It Works (High-Level)
User Input → Embedding

Takes user text, converts it into a vector (ASCII values scaled to floats).
Context & Attention

Persistent memory tokens + recent user messages + current user embedding are concatenated.
A short-term attention block processes the combined sequence.
“Surprise” Calculation

The final user embedding from the attention output is tested for its “surprise” factor.
We do a tiny forward/backward pass in the DeepNeuralMemory to measure gradient magnitude.
Memory Buffer & Update

If surprise ≥ threshold, the embedding is stored in chunk_buffer.
When the buffer is full, run an MSE loss update to adjust memory weights.
LLM Call

The agent then calls an external LLM (by default, groq.chat.completions.create) using the prepared prompt containing the short-term summary and memory log.
Response

The agent’s final text is displayed in the console under [TitansAgent].
7. Customization
Threshold Tuning
surprise_threshold: Higher values → fewer updates, focusing on only the most “surprising” inputs.
Memory Depth
num_layers for deeper or shallower long-term memory.
Optimizer Hyperparameters
Learning rate lr, momentum, and weight_decay to control how quickly memory updates and how much it forgets.
Embedding Function
_embed_text(): Currently ASCII-based; you can integrate a real NLP embedding.
8. Known Limitations
Limited Context

The agent only attends to the last short_term_capacity messages + persistent tokens. For extremely long dialogues, older messages can be overshadowed unless properly updated in the MLP.
Ad-Hoc Surprise Calculation

Using gradient magnitude is a heuristic. More advanced gating or a better measure of “importance” might work better in practice.
Token-Level vs. Message-Level

This script does test-time training at the message level. Fine-grained token-level updates may require more sophisticated chunking strategies.
Security & Privacy

Any text typed in is used to update the memory, potentially storing sensitive data in weights. Consider how you manage or clear that data if needed.
9. Further Directions
Retrieval-Augmented Memory

Hybrid approach: maintain a vector datastore for large-scale recall and rely on the MLP for abstracted knowledge.
Transformer / RNN Switch

The short-term memory block could be replaced with different architectures (e.g., a sliding window Transformer, linear RNN, or SSM-based block).
Integration with Larger Systems

This agent can be wrapped in a larger framework that processes user contexts, multi-modal inputs, or advanced conversation flows.
10. Contributing
Feel free to open pull requests or file issues for:

Refactoring the adaptive gating to handle different gating strategies.
Integrating more advanced front-end embeddings or LLM endpoints.
Optimizing the memory update logic (e.g., parallelizing chunk updates).
11. License
(Add your license information here if any. For example, MIT, Apache 2.0, or Proprietary. Otherwise, omit this section.)

Acknowledgments
Titans Paper & Inspiration: This code is inspired by the research on “Titans: Learning to Memorize at Test Time.”
PyTorch: for autograd, optimizers, and modules.
Any Additional Tools or resources that influenced this approach.
Contact
Author: [Your Name] – [Your Email or LinkedIn]
Project Repository: (If you host it on GitHub/GitLab, add the link.)
Thank you for checking out this README. Feel free to experiment, modify, and extend the Titans-based agent to suit your own needs!
