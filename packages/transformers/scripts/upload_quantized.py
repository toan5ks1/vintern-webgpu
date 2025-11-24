from huggingface_hub import HfApi

repo_id = "toan5ks1/Vintern-1B-v3_5-ONNX"
files_to_upload = [
    ("temp_quant_output/decoder_model_merged_q4.onnx", "onnx/decoder_model_merged_q4.onnx")
]

api = HfApi()

for file_path, path_in_repo in files_to_upload:
    print(f"Uploading {file_path} to {repo_id} as {path_in_repo}...")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload quantized decoder model (q4)"
        )
        print(f"Successfully uploaded {path_in_repo}")
    except Exception as e:
        print(f"Error uploading {path_in_repo}: {e}")
