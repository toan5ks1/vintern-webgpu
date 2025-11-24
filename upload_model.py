from huggingface_hub import HfApi
import sys

api = HfApi()

repo_id = "toan5ks1/Vintern-1B-v3_5-ONNX"
local_path = "models_1/vintern-1b-v3_5-onnx/onnx/decoder_model_merged_quantized.onnx"
path_in_repo = "onnx/decoder_model_merged_quantized.onnx"

print(f"Uploading {local_path} to {repo_id}/{path_in_repo}...")

try:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model"
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}")
    sys.exit(1)
