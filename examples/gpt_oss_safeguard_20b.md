```python
!pip install ./cacheflow-0.1.0-py3-none-any.whl
```

    Processing ./cacheflow-0.1.0-py3-none-any.whl
    Requirement already satisfied: torch>=2.1 in /usr/local/lib/python3.12/dist-packages (from cacheflow==0.1.0) (2.10.0+cu128)
    Requirement already satisfied: safetensors>=0.4 in /usr/local/lib/python3.12/dist-packages (from cacheflow==0.1.0) (0.7.0)
    Requirement already satisfied: huggingface-hub>=1.10.2 in /usr/local/lib/python3.12/dist-packages (from cacheflow==0.1.0) (1.11.0)
    Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=1.10.2->cacheflow==0.1.0) (3.29.0)
 ==Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=1.10.2->cacheflow==0.1.0) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=1.10.2->cacheflow==0.1.0) (2.20.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub>=1.10.2->cacheflow==0.1.0) (0.1.2)
    Installing collected packages: cacheflow
    Successfully installed cacheflow-0.1.0

```python
from huggingface_hub import snapshot_download

model_name = "openai/gpt-oss-safeguard-20b"
# model_name="deepseek-ai/DeepSeek-V2-Lite"
model_path = "./weights/granite/"

snapshot_download(model_name, local_dir=model_path)
```

    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:93: UserWarning:
    The secret`HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(

    Downloading (incomplete total...): 0.00B [00:00, ?B/s]

    Fetching 15 files:   0%|          | 0/15 [00:00<?, ?it/s]

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

    '/content/weights/granite'

```python
import os
import json
from safetensors import safe_open
from safetensors.torch import save_file


def separate_safetensors_smart(input_path, output_directory):
    # 1. Create the output directory
    os.makedirs(output_directory, exist_ok=True)

    new_weight_map = {}
    total_size = 0
    files_to_process = []
    input_dir = os.path.dirname(input_path)

    # 2. Figure out if the input is a single file or an index.json
    if input_path.endswith(".json"):
        print(f"Detected index JSON. Reading {input_path}...")
        with open(input_path, "r") as f:
            index_data = json.load(f)

        # Look at the old weight map to find out what shard files exist
        old_weight_map = index_data.get("weight_map", {})

        # Get a list of unique shard files (e.g. model-00001-of-00003.safetensors)
        unique_shards = set(old_weight_map.values())

        # Build the full paths to those shards
        for shard in unique_shards:
            shard_path = os.path.join(input_dir, shard) if input_dir else shard
            files_to_process.append(shard_path)

        print(f"Found {len(files_to_process)} shard file(s) to unpack.")

    elif input_path.endswith(".safetensors"):
        print(f"Detected single safetensors file: {input_path}")
        files_to_process = [input_path]
    else:
        raise ValueError("Input file must be a .safetensors or .index.json file")

    # 3. Process every file we found
    tensor_count = 0

    for file_path in files_to_process:
        print(f"\nOpening {os.path.basename(file_path)}...")

        # Open lazily
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in keys:
                tensor_count += 1

                # Extract only this specific layer into RAM
                tensor = f.get_tensor(key)

                # Create exact filename for this layer
                filename = f"{key}.safetensors"
                output_path = os.path.join(output_directory, filename)

                # Save it to the new folder
                save_file({key: tensor}, output_path)

                # Track size and map it for the new index
                file_size = os.path.getsize(output_path)
                total_size += file_size
                new_weight_map[key] = filename

                print(f"  Saved: {filename}")

    # 4. Generate the NEW model.safetensors.index.json
    print("\nAll layers separated. Generating new model.safetensors.index.json...")

    new_index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }

    index_out_path = os.path.join(output_directory, "model.safetensors.index.json")

    with open(index_out_path, "w") as index_file:
        json.dump(new_index_data, index_file, indent=2)

    print(f"\n--- SUCCESS ---")
    print(f"Total separate tensors saved: {tensor_count}")
    print(f"Total model size: {total_size / (1024**3):.2f} GB")
    print(f"New index saved to: {index_out_path}")


# ==========================================
# CONFIGURATION - GPT-OSS PATHS
# ==========================================

INPUT_PATH = "./weights/granite/model.safetensors.index.json"
OUTPUT_DIR = "./weights/granite/"

separate_safetensors_smart(INPUT_PATH, OUTPUT_DIR)
```

    Detected index JSON. Reading ./weights/granite/model.safetensors.index.json...
    Found 3 shard file(s) to unpack.

    Opening model-00001-of-00002.safetensors...
      Saved: model.layers.18.mlp.experts.down_proj_bias.safetensors
      Saved: model.layers.18.mlp.experts.down_proj_blocks.safetensors
    All layers separated. Generating new model.safetensors.index.json...

    --- SUCCESS ---
    Total separate tensors saved: 459
    Total model size: 12.82 GB
    New index saved to: ./weights/granite/model.safetensors.index.json

```python
from transformers import AutoTokenizer
import torch
from cacheflow import initialize_model


model_path = "./weights/granite/"
loaded = initialize_model(model_path)
print(loaded.device, loaded.dtype)


tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = loaded.config.pad_token_id
```

    cuda torch.float16

```python
torch.cuda.reset_peak_memory_stats()
question = "What is the capital of Bangladesh?"
prompt = f"Question: {question}\nAnswer:"
# prompt = f"write 10 line poem about python language"

result = loaded.generate_text(
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=10,
    min_new_tokens=8,
    temperature=0.5,
    top_k=10,
    eos_token_id=loaded.config.eos_token_id,
    repetition_penalty=1.15,
    block_special_tokens=True,
    return_raw=True,
)

print("generated_token_ids:", result["token_ids"])
print("decoded_raw:", repr(result["raw_text"]))
print("decoded:", repr(result["text"]))
```

    generated_token_ids: [43278, 751, 27968, 93028, 36598, 10825, 41478, 16696, 8443, 540]
    decoded_raw: 'direction publicSWPW biological stack cycles identity abs at'
    decoded: 'direction publicSWPW biological stack cycles identity abs at'

```python
print(
    "after warmup alloc/res:",
    torch.cuda.memory_allocated() / 1e9,
    torch.cuda.memory_reserved() / 1e9,
)
print("peak alloc:", torch.cuda.max_memory_allocated() / 1e9)
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    curr_mem = torch.cuda.memory_allocated() / (1024**2)
    print(f"🔥 Peak VRAM Allocated:   {peak_mem:.2f} MB")
    print(f"   Current VRAM Allocated: {curr_mem:.2f} MB")
```

    after warmup alloc/res: 6.216499712 14.189330432
    peak alloc: 6.632850432
    🔥 Peak VRAM Allocated:   6325.58 MB
       Current VRAM Allocated: 5928.52 MB
