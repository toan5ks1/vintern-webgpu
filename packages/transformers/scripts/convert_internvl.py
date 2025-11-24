
import argparse
import os
import shutil
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from optimum.exporters.onnx import main_export
import json

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

class InternVLVisionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.mlp1 = model.mlp1
        self.downsample_ratio = model.downsample_ratio
        self.select_layer = model.select_layer

    def forward(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        
        # Drop CLS token
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    # Load original model
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    output_model_folder = args.output_dir
    os.makedirs(output_model_folder, exist_ok=True)

    # 1. Export Vision Model
    print("Exporting Vision Model...")
    vision_wrapper = InternVLVisionWrapper(model)
    vision_wrapper.eval()

    # Create dummy input
    # InternViT-300M-448px uses 448x448 images
    dummy_input = torch.randn(1, 3, 448, 448)
    
    vision_onnx_path = os.path.join(output_model_folder, "vision_encoder.onnx")
    torch.onnx.export(
        vision_wrapper,
        dummy_input,
        vision_onnx_path,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )
    print(f"Vision model exported to {vision_onnx_path}")

    # 2. Export Language Model (Qwen2)
    print("Exporting Language Model...")
    
    # Load tokenizer early
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # 2a. Export Embedding Layer (embed_tokens)
    print("Exporting Embedding Layer...")
    embed_tokens = model.language_model.model.embed_tokens
    embed_tokens_path = os.path.join(output_model_folder, "embed_tokens.onnx")
    dummy_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    torch.onnx.export(
        embed_tokens,
        dummy_input_ids,
        embed_tokens_path,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )

    # Save the language model config and weights temporarily to export with optimum
    lm_path = os.path.join(output_model_folder, "_tmp_lm")
    model.language_model.save_pretrained(lm_path)
    tokenizer.save_pretrained(lm_path) # Save tokenizer with LM

    # Use optimum to export the language model
    # We use 'text-generation-with-past' task
    # We need to ensure inputs_embeds is exported.
    # We can create a custom config or patch the existing one.
    from optimum.exporters.onnx.model_configs import TextDecoderOnnxConfig
    from optimum.utils import NormalizedTextConfig
    
    try:
        from optimum.exporters.onnx.model_configs import Qwen2OnnxConfig
        ParentConfig = Qwen2OnnxConfig
    except ImportError:
        ParentConfig = TextDecoderOnnxConfig

    class CustomQwen2OnnxConfig(ParentConfig):
        NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
        def __init__(self, config, task="text-generation", use_past=False, use_past_in_inputs=False, **kwargs):
            super().__init__(config, task=task, use_past=use_past, use_past_in_inputs=use_past_in_inputs, **kwargs)
            self._include_inputs_embeds = True

        @property
        def inputs(self):
            inputs = super().inputs
            if self._include_inputs_embeds:
                if "input_ids" in inputs:
                    del inputs["input_ids"]
                inputs["inputs_embeds"] = {0: "batch_size", 1: "sequence_length"}
            return inputs

        def generate_dummy_inputs(self, framework="pt", **kwargs):
            self._include_inputs_embeds = False
            dummy_inputs = super().generate_dummy_inputs(framework=framework, **kwargs)
            self._include_inputs_embeds = True
            
            batch_size = dummy_inputs["input_ids"].shape[0]
            seq_len = dummy_inputs["input_ids"].shape[1]
            hidden_size = self._config.hidden_size
            dummy_inputs["inputs_embeds"] = torch.randn(batch_size, seq_len, hidden_size)
            
            if "input_ids" in dummy_inputs:
                del dummy_inputs["input_ids"]
                
            return dummy_inputs

    # Load config for the wrapper
    lm_config = AutoConfig.from_pretrained(lm_path)
    onnx_config = CustomQwen2OnnxConfig(
        config=NormalizedTextConfig(lm_config),
        task="text-generation",
        use_past=True,
        use_past_in_inputs=True,
    )

    main_export(
        model_name_or_path=lm_path,
        output=output_model_folder,
        task="text-generation-with-past",
        opset=14,
        no_post_process=True,
        custom_onnx_configs={"model": onnx_config}
    )
    
    # Rename model.onnx to decoder_model_merged.onnx
    # Optimum exports 'model.onnx' for this task
    if os.path.exists(os.path.join(output_model_folder, "model.onnx")):
        os.rename(
            os.path.join(output_model_folder, "model.onnx"),
            os.path.join(output_model_folder, "decoder_model_merged.onnx")
        )
    
    # Cleanup tmp LM
    shutil.rmtree(lm_path)

    # 3. Handle Tokenizer and Configs
    print("Saving tokenizer and configs...")
    tokenizer.save_pretrained(output_model_folder)
    
    # Calculate image_token_index
    # InternVL uses <IMG_CONTEXT> as the placeholder
    image_token_index = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    print(f"Image token index: {image_token_index}")

    # Save preprocessor config
    # We need to construct a preprocessor_config.json that transformers.js can use
    # InternVL uses a specific preprocessing. We can mimic what's needed.
    # Usually it's just Resize + Normalize.
    preprocessor_config = {
        "do_normalize": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "size": {"height": 448, "width": 448},
        "resample": 3, # BICUBIC
        "image_processor_type": "InternVLImageProcessor" # We might need to map this in JS
    }
    with open(os.path.join(output_model_folder, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f, indent=2)

    # Save main config
    model.config.image_token_index = image_token_index
    model.config.save_pretrained(output_model_folder)

    if args.quantize:
        print("Quantizing models...")
        from optimum.onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Quantize vision model
        quantize_dynamic(
            vision_onnx_path,
            os.path.join(output_model_folder, "vision_model_quantized.onnx"),
            weight_type=QuantType.QUInt8
        )
        os.remove(vision_onnx_path)
        os.rename(os.path.join(output_model_folder, "vision_model_quantized.onnx"), vision_onnx_path)

        # Quantize LM (decoder_model.onnx and decoder_with_past_model.onnx)
        for model_name in ["model.onnx", "model_quantized.onnx"]: # Check what optimum outputs
             # Optimum usually outputs 'model.onnx' for merged, or decoder_model.onnx etc.
             # We'll just check what files exist and quantize them if they are ONNX
             pass
        
        # Actually, let's just use the existing quantize script or function if possible, 
        # but for now simple dynamic quantization on everything ending in .onnx
        
        for file in os.listdir(output_model_folder):
            if file.endswith(".onnx") and "vision" not in file: # Vision already done
                path = os.path.join(output_model_folder, file)
                quant_path = os.path.join(output_model_folder, f"quant_{file}")
                print(f"Quantizing {file}...")
                quantize_dynamic(
                    path,
                    quant_path,
                    weight_type=QuantType.QUInt8
                )
                os.remove(path)
                os.rename(quant_path, path)

    print("Conversion complete!")

if __name__ == "__main__":
    main()
