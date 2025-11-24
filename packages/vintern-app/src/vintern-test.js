
import {
    AutoTokenizer,
    InternVLChatModel,
    RawImage,
    AutoProcessor,
    Tensor
} from '@huggingface/transformers';

async function main() {
    const model_id = '/Users/toannguyen/Documents/app/models/vintern-1b-v3_5-onnx';
    
    console.log(`Loading model from ${model_id}...`);
    const tokenizer = await AutoTokenizer.from_pretrained(model_id);
    const model = await InternVLChatModel.from_pretrained(model_id, {
        dtype: 'fp32', // using fp32 to match available ONNX files
        device: 'cpu', // Node.js doesn't support webgpu
    });

    console.log("Model loaded.");

    // Load image
    const url = '/Users/toannguyen/Documents/app/packages/vintern-app/public/images/bb.jpg';
    const image = await RawImage.fromURL(url);

    // Preprocess image
    // We manually preprocess for now as we didn't implement InternVLImageProcessor in JS yet
    // But we can use AutoProcessor if we mapped it or if generic ImageProcessor works.
    // Our config has "image_processor_type": "InternVLImageProcessor".
    // transformers.js might fallback to generic or fail.
    // Let's try to manually create pixel_values.
    
    // InternVL preprocessing: Resize to 448x448, Normalize.
    // We can use the processor if it works, or manual.
    // Let's try to use a generic processor with the config we saved.
    // We saved preprocessor_config.json.
    
    // If AutoProcessor fails, we'll do manual.
    let pixel_values;
    try {
        const processor = await AutoProcessor.from_pretrained(model_id);
        const vision_inputs = await processor(image);
        pixel_values = vision_inputs.pixel_values;
    } catch (e) {
        console.log("AutoProcessor failed, trying manual preprocessing:", e);
        // Manual preprocessing
        // Resize to 448x448
        const resized = await image.resize(448, 448);
        
        // Convert to tensor and normalize
        const tensor = new Tensor('float32', new Float32Array(1 * 3 * 448 * 448), [1, 3, 448, 448]);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        // RawImage data is usually RGBA or RGB. transformers.js RawImage handles it.
        // We can use `toTensor` if available, or get data.
        // resized.data is Uint8Array (H * W * 4) usually.
        const data = resized.data;
        const width = resized.width;
        const height = resized.height;
        const channels = resized.channels;
        
        console.log(`Image info: ${width}x${height}x${channels}`);

        for (let i = 0; i < width * height; ++i) {
            const offset = channels * i;
            const r = data[offset] / 255.0;
            const g = data[offset + 1] / 255.0;
            const b = data[offset + 2] / 255.0;
            
            tensor.data[i] = (r - mean[0]) / std[0]; // R
            tensor.data[width * height + i] = (g - mean[1]) / std[1]; // G
            tensor.data[2 * width * height + i] = (b - mean[2]) / std[2]; // B
        }
        pixel_values = tensor;
    }

    const question = "Trích xuất thông tin và trả về dạng JSON.";
    const messages = [
        { role: 'user', content: `<image>\n${question}` }
    ];
    
    // Apply chat template
    // We need to ensure the tokenizer has the template.
    // transformers.js AutoTokenizer should load it.
    let prompt = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
    
    // Replace <image> with 256 <IMG_CONTEXT> tokens
    const image_tokens = "<IMG_CONTEXT>".repeat(256);
    prompt = prompt.replace("<image>", image_tokens);
    
    console.log("Prompt:", prompt.slice(0, 100) + "..." + prompt.slice(-100));

    const inputs = tokenizer(prompt);
    inputs.pixel_values = pixel_values;

    // Debug: Check embed_tokens
    if (model.sessions['embed_tokens']) {
        const dummy_ids = new Tensor('int64', [151667n], [1, 1]); // <IMG_CONTEXT>
        const embeds = await model.sessions['embed_tokens'].run({ input_ids: dummy_ids });
        console.log("Embeddings for <IMG_CONTEXT>:", embeds.inputs_embeds.data.slice(0, 10));
    }

    // Debug: Check pixel_values for NaN
    let hasNaN = false;
    for (let i = 0; i < pixel_values.data.length; ++i) {
        if (isNaN(pixel_values.data[i])) {
            hasNaN = true;
            break;
        }
    }
    console.log("Pixel values has NaN:", hasNaN);

    // Debug: Check vision_encoder with dummy input
    const dummy_pixel_values = new Tensor('float32', new Float32Array(1 * 3 * 448 * 448).fill(0.5), [1, 3, 448, 448]);
    const dummy_out = await model.encode_image({ pixel_values: dummy_pixel_values });
    console.log("Dummy vision output:", dummy_out.data.slice(0, 10));

    // Debug: Check vision_encoder
    const vision_out = await model.encode_image({ pixel_values });
    console.log("Vision output dims:", vision_out.dims);
    console.log("Vision output data (first 10):", vision_out.data.slice(0, 10));

    console.log("Generating with image...");
    const outputs = await model.generate({
        ...inputs,
        max_new_tokens: 100,
    });

    const decoded = tokenizer.batch_decode(outputs, { skip_special_tokens: true });
    console.log("Output:", decoded);
}

main().catch(console.error);
