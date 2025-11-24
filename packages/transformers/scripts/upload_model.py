import os
from huggingface_hub import HfApi, create_repo

model_path = "/Users/toannguyen/Documents/app/models/vintern-1b-v3_5-onnx"
repo_id = "toan5ks1/Vintern-1B-v3_5-ONNX"

api = HfApi()

print(f"Creating repository {repo_id} if it doesn't exist...")
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Error creating repo: {e}")

print(f"Uploading files from {model_path} to {repo_id}...")
try:
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload converted ONNX model"
    )
    print("Upload complete!")
except Exception as e:
    print(f"Error uploading files: {e}")
