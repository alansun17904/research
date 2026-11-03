GOAL. Your goal is to modify the computational graph of a pretrained model to maximize its performance (measured in language model perplexity on WikiText). The model whose architecture you will modify is`EleutherAI/pythia-70m`. You are not allowed to edit the weights of the pretrained model at all. After each newly proposed architecture, you are allowed to run forward passes on the validation dataset view the new model's performance and also specific generated examples. It would help to also examine the weights of the model to try and figure out what the components are doing. 

ARCHITECTURE CONSTRAINTS. Here are the architecture modifications that you are allowed to make: 
1. Delete an attention head (i.e. zero out all of the weights here). 
2. Loop an attention head arbitrarily many times. 
3. Loop an MLP arbitrarily many times.
4. Loop a subset of layers arbitrarily many times.
There is no cap on the repetition counts. For example, if you know that there is an induction head in layers 1-2 (say specifically in attention head i, j), then you could loop a system of these attention heads (1.i -> 2.j -> 1.i -> ...) to create higher order attention heads. As another example, an MLP can be looped to perform hierarchical retrieval: on the first iteration it maps key i -> value j, then on the second iteration it maps value j -> value k and implement some sort of tree-based search. At the moment, ignore any efficiency concerns. 

TURN BY TURN ETIQUETTE. Before each architecture modification, provide a justification for why this particular decision was made. In creating this justification, you are allowed to reference specific examples from the validation dataset and also the existing architecture from all the previous rounds. You should write these justifications in a document. This will be used by future versions of you as a log. 

EDITABLE PYTHIA 70M. `model.py` is a small PyTorch implementation of the pinned `EleutherAI/pythia-70m` checkpoint. `graphs/baseline.json` reproduces its original six layers. `graphs/example.json` demonstrates the four allowed operations; it is an example, with no claim of improved WikiText perplexity.

FILES.
- `assets/pythia-70m/`: downloaded original weights, config, tokenizer, model card, and a SHA-256 provenance manifest. Assets and `.venv/` are git-ignored.
- `reference/modeling_gpt_neox.py`: untouched official Transformers v4.57.1 source, with its Apache-2.0 `LICENSE` and source URL/hash manifest.
- `model.py`: architecture, JSON validation, and strict checkpoint loading.
- `evaluate.py --graph PATH`: evaluates the JSON model through the existing token-counted WikiText validation evaluator.
- `test_model.py`: real-checkpoint parity and focused operation/validation checks. `verification.json` records the check results and environment.

CHECKPOINT LOADING. The downloader uses the model revision pinned in `evaluate.py`. All 94 checkpoint tensors map by their original names with `strict=True`: 76 parameter tensors and 18 legacy buffers. Checkpoint/config checksums are verified before loading. There are 70,426,624 unique stored parameters, including inactive head slices; repeated references allocate no additional parameters. Loaded parameters have gradients disabled. No training or weight-update code is provided.

JSON ARCHITECTURE ENCODING. Read the explicit execution semantics below. The JSON contains structure and original checkpoint identities, without tensor values. Propose architectures by editing JSON files, keeping the original checkpoint tensors unchanged. After recording a candidate's justification, copy `graphs/baseline.json` (or the best measured graph) to a new file such as `graphs/candidate-001.json`. Keep earlier candidates so each result can be reproduced. `graphs/example.json` demonstrates all four operations; it is not a measured improvement.

The JSON has exactly five top-level fields:

- `format`: keep `"pythia-70m-v1"`.
- `checkpoint`: keep the baseline's model name and pinned revision unchanged.
- `weights`: keep the baseline's embedding, final norm, and output identities.
- `layers`: keep six definitions in original layer order, with `weights` naming `gpt_neox.layers.0` through `gpt_neox.layers.5`. Each definition has exactly `weights`, `heads`, and `mlp_repeat`.
- `sequence`: a nonempty list of execution blocks, visited in the given order.

All layer indices (`0..5`) and head indices (`0..7`) are zero-based. Encode the four allowed operations as follows:

1. Delete a head: set `layers[L].heads[H]` to `0`. Each layer's `heads` array must contain eight nonnegative integer counts. Deletion masks the head's entire contribution, including its Q/K/V biases, without editing stored weights. The shared attention output bias remains once per full layer visit.
2. Loop a head: set `layers[L].heads[H]` to the total application count. `1` is the original behavior; `3` means the initial application plus two recomputed applications on feedback from the updated attention branch. Extra applications run in ascending head-index order within a layer.
3. Loop an MLP: set `layers[L].mlp_repeat` to a positive integer total count. `1` is the original behavior; `3` adds two recomputed feedback applications. Attention and MLP branches remain parallel within a full layer visit.
4. Loop a layer subset: add a `sequence` block such as `{"layers": [1, 3], "repeat": 2}`. It executes `1, 3, 1, 3`, using the same original tensors each time. A layer's head and MLP counts apply on every visit. Selected layer indices need not be contiguous and may repeat. Keep the ordinary traversal where intended; for example:

   ```json
   "sequence": [
     {"layers": [0, 1, 2, 3, 4, 5], "repeat": 1},
     {"layers": [1, 3], "repeat": 2}
   ]
   ```

For cross-layer head composition, a block can instead contain `heads` pairs: `{"heads": [[1, 2], [2, 3]], "repeat": 2}` executes head `1.2`, then `2.3`, then repeats that path. Each pair adds one recomputed head contribution to the current residual stream, with its original input layer norm, but no MLP or shared attention output bias. A head whose layer count is `0` is skipped; positive layer counts do not multiply head-only applications. Use the block's `repeat` or repeated pairs to loop this path. Each block contains exactly one of `layers` or `heads`, plus a positive integer `repeat`.

Repetitions reuse weights and fixed token positions; they do not multiply a cached output or create new parameters. Counts have no upper cap. Large counts execute literally and can take correspondingly long; loops are not expanded into copies of modules or tensors. Use JSON integers, not booleans or fractions; duplicate keys, extra fields, invalid indices, negative counts, and non-finite numbers are rejected.

EXPLICIT EXECUTION SEMANTICS. These are the implementation's starting choices, open to review. Each repeat reuses the selected original tensors and recomputes its operation on updated hidden states. Token positions stay fixed across repeated applications.

For a layer input `x`, let `C_h(z)` be head `h`'s causal attention on `input_layernorm(z)`, followed by that head's columns of the original attention output projection, excluding its shared bias `b`. Let `M(z)` be the full original MLP, including both linear biases and exact GELU, on `post_attention_layernorm(z)`.

1. Start the attention delta with `a = b + sum(C_h(x))` over active heads. For each extra head application, update `a = a + C_h(x + a)`. Within a layer, extras run by ascending original head index. A count of three means one initial application plus two recomputed applications, rather than multiplication of an initial output.
2. Start the MLP delta with `m = M(x)`. Each extra MLP application updates `m = m + M(x + m)`.
3. Return `m + a + x`. The two recurrent branches remain parallel, matching the original residual behavior when every count is one.
4. A head-only sequence entry instead updates the current stream directly: `x = x + C_h(x)`. It adds no shared attention output bias or MLP. This makes cross-layer head composition explicit in the JSON order.

Complete layer-subset loops feed each layer's result into the next listed layer, then repeat that block. All occurrences refer to the same six stored modules. The example deletes layer 0/head 0, applies layer 1/head 1 twice per layer visit, applies layer 2's MLP twice per visit, adds two passes through layers `[1, 3]`, and adds two passes through the head-only path `[[1, 1], [2, 2]]`.

The implementation retains the checkpoint's biases, LayerNorm epsilon, fused per-head Q/K/V layout, causal masking, partial rotary embedding, exact GELU, parallel residuals, final norm, and untied output head. For parity with the official implementation, rotary frequencies are regenerated in FP32 from the config; the legacy checkpoint's rounded FP16 frequency buffers are also retained and strictly loaded. The current path supports unpadded full sequences and next-token loss, without KV caching or a generation framework.

PERFORMING MEASUREMENTS. Read `AGENT.md` and `evaluate.py` before changing code. The evaluator uses `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, split `validation`. It pins the model and dataset revisions in its source and takes the first 100,000 tokens by default, in dataset order. Text rows are joined with two newlines; tokenization adds no special tokens or padding. The first token is context only, yielding 99,999 scored next-token targets for the default sample. Pretraining overlap has not been audited.

Evaluation uses overlapping windows within the model's context limit and a half-context stride. For the starting model, these are 2,048 and 1,024 tokens. Each target is scored once, excluding context-only labels. The output JSON reports the actual input and scored-token counts, revisions, window settings, total and mean negative log-likelihood in nats, and perplexity: `perplexity = exp(mean_nll_nats)`. Lower perplexity is better.

MINIMAL WORKFLOW.

1. Inspect this file, `model.py`, `evaluate.py`, and the existing graphs. Resolve implementation decisions before relying on them.
2. Prepare the environment and assets using the setup commands below. Run the offline counting check, `.venv/bin/python evaluate.py --self-check`, and the implementation checks, `.venv/bin/python -m unittest -v`.
3. Establish and save an unmodified Pythia 70M baseline using `--graph graphs/baseline.json`, following the evaluation commands below inside a Slurm compute job.
4. Record a justification before each architecture modification, encode the candidate using the four allowed operations, validate its JSON, and evaluate it with `--graph PATH`. Preserve the same pinned starting checkpoint, tokenizer, validation sample, preprocessing, context length, stride, target accounting, device, and environment. Record the exact operations and their implementation semantics for each candidate.
5. Compare measured perplexity with the baseline and the best result so far. Use those results to choose subsequent candidates, retaining reproducible configurations and reporting failures as well as improvements.

CLUSTER OPERATIONS. Adapted from `/Users/alansun/Documents/research/non-linear/GEMINI.md`.

ACCESS AND EXECUTION.

- Run experiments on the group's compute cluster through Slurm. Access the Slurm orchestrator with `ssh kennel`.
- Keep the agent on the local machine and sync code to the cluster. Use `slurm-login.csail.mit.edu` only for job submission; `login.csail.mit.edu` is for lightweight interactive use. Do not run the agent on either login host; the Slurm login host also kills editors. If work must happen beside cluster data, ask the user to allocate a compute node and connect there.
- Connections to `*.csail.mit.edu` use `jump.csail.mit.edu` and require Duo. The user must establish the connection manually before cluster work. Check an existing shared connection with `ssh -O check jump.csail.mit.edu`; `Master running` indicates it is available.
- If SSH fails, stop. Do not retry, switch CSAIL hosts, or fall back to password or public-key authentication. Repeated failures trigger sshguard blocks for everyone sharing the IP address. Ask the user to run `ssh login.csail.mit.edu`, complete Duo, and leave the session open. Report expired Kerberos tickets or AFS tokens and wait; do not retry.

STORAGE.

- Use `/data/cl/u/awsun` for cluster development, code, environments, data, checkpoints, and results that must persist.
- Use `/data/cl/scratch/awsun` only for disposable temporary files; they may be deleted at any time. This is NFS, not node-local storage.
- Do not use the AFS home (`~`) for code, data, checkpoints, environments, or job files. AFS tokens expire and are unavailable inside Slurm jobs.
- For data-intensive jobs, stage data to the compute node's local `/scratch` and read it there to reduce shared NFS traffic.
- Stay within the project directory; do not scan or bulk-write a home directory or its parents. Use local `/tmp` for sandboxes, caches, and build artifacts.

PARTITIONS. Select the partition with `--partition`.

| Partition | Use | GPU limit | Maximum job time | Preemption |
| --- | --- | --- | --- | --- |
| `debug` | Quick debugging | 2 GPUs at a time | 2 hours | Never preempted |
| `standard` (default) | General jobs | 4 GPUs at a time | 24 hours | Not specified in source |
| `long` | Very large jobs | No limit | 7 days | Expect frequent preemption |
| `sweep` | Sweeps intended to saturate available compute | Same as `long` | 7 days | Immediate termination on preemption, with no cleanup window |

Even `debug` allocations may take 1–2 minutes while another job wraps up.

FILES AND SETUP. The local environment and assets have already been installed/downloaded. To recreate them in the synced project directory, run the following inside a Slurm compute job:

```bash
export HF_HOME=/data/cl/u/awsun/.cache/huggingface
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python fetch_assets.py
.venv/bin/python -m unittest -v
.venv/bin/python evaluate.py --self-check
```

`--graph` requires the checkpoint, tokenizer, config, and provenance manifest under `assets/pythia-70m/`; `fetch_assets.py` downloads them. Keep the pinned dependencies from `requirements.txt`.

RUNNING JSON GRAPH EVALUATIONS. Use the existing CLI with `--graph PATH` to evaluate a JSON architecture with its local original tensors; no new runner is needed. Omitting `--graph` evaluates the untouched official model; include it for both the JSON baseline and every candidate to compare the same executor.

Follow the cluster access, storage, and partition rules above. Run the following commands inside a Slurm compute job, from the synced project directory.

Before scoring, run the offline counting check and validate the candidate JSON without a WikiText evaluation:

```bash
.venv/bin/python evaluate.py --self-check
.venv/bin/python -c 'from model import read_graph; read_graph("graphs/candidate-001.json"); print("Graph valid")'
```

In the allocated compute job, save baseline and candidate results separately:

```bash
export HF_HOME=/data/cl/u/awsun/.cache/huggingface
mkdir -p results
.venv/bin/python evaluate.py --graph graphs/baseline.json --max-tokens 100000 --device cuda > results/baseline.json
.venv/bin/python evaluate.py --graph graphs/candidate-001.json --max-tokens 100000 --device cuda > results/candidate-001.json
```

Use a new result filename for each candidate. `--device cpu` is also supported. For a shorter trial, reduce `--max-tokens` (minimum `2`) for both baseline and candidate; report those measurements separately from the full 100,000-token comparison. Keep the device and environment consistent across comparisons.

The result JSON includes the executed `graph`, `unique_parameters`, pinned model/dataset revisions, actual token counts, window settings, NLL, and perplexity. Confirm the command succeeded and the output is complete JSON before logging a measurement. Compare `perplexity` and `mean_nll_nats` on the same scored targets; lower is better. Report absolute perplexity change as `candidate - baseline` and relative change as `100 * (candidate / baseline - 1)` percent; negative values are improvements. Record the candidate path, justification, command, result path, and comparison with both the baseline and the best result so far in the experiment log. The Python interface remains `evaluate.evaluate(model, input_ids)` for a supplied model and token IDs. The CLI reports loss/perplexity, not generated examples. JSON validation and `--self-check` alone do not measure model performance.

RESULTS TO REPORT.

- Baseline and candidate perplexity, mean NLL, actual scored-token count, and absolute and relative perplexity change from baseline.
- The exact candidate configuration: affected heads, MLPs, or layer subsets, repetition counts, deletion choices, and resolved operation semantics.
- Reproduction instructions: code/configuration, model and dataset revisions, environment/package versions, evaluation commands, and saved result paths.
- The best measured configuration, unsuccessful candidates, limitations, and any remaining open decisions. Clearly distinguish completed evaluations from plans, counting checks, and untested hypotheses.

VALIDATION SCOPE. The parity check compares CPU FP32 logits over 257 deterministic text tokens against the official eager implementation loaded from the same real checkpoint. Absolute tolerances are `2e-4` for logits, `1e-4` for per-target log probabilities, and `1e-5` for mean NLL; relative tolerances are zero. `verification.json` records the measured errors. This is a synthetic implementation check, not a WikiText perplexity result or a guarantee of identical outputs across devices/dtypes.

Other checks cover all four operations, cross-layer head composition, duplicate references without parameter copies, causal masking, complete checkpoint mapping, checksums, malformed JSON, uncapped integer counts, weighted partial evaluation windows, and a synthetic `--graph` CLI run that fails if the untouched official model is accidentally reloaded. Full WikiText evaluation and architecture search have not been run; pretraining overlap with WikiText has not been audited.
