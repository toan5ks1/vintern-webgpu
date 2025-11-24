import onnx
import os
import sys

def verify_model(path):
    print(f"Verifying {path}...")
    try:
        # Just load the model to check if protobuf parsing works
        # load_model only loads the protobuf, it doesn't check graph validity fully but is enough for "protobuf parsing failed"
        model = onnx.load_model(path, load_external_data=False)
        print(f"Successfully loaded {path}")
        return True
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return False

model_dir = "packages/vintern-app/public/models/vintern-1b-v3_5-onnx"
files = [
    "embed_tokens.onnx",
    "vision_encoder.onnx",
    "decoder_model_merged_q8.onnx"
]

all_good = True
for f in files:
    path = os.path.join(model_dir, f)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        all_good = False
        continue
    
    if not verify_model(path):
        all_good = False

if all_good:
    print("All models verified successfully.")
    sys.exit(0)
else:
    print("Some models failed verification.")
    sys.exit(1)
