import {
    AutoTokenizer,
    InternVLChatModel,
    RawImage,
    AutoProcessor,
    env,
    Tensor,
    TextStreamer
} from '@huggingface/transformers';

// Configure environment
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let model = null;
let tokenizer = null;
let processor = null;

self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    try {
        if (type === 'load') {
            await loadModel(data.modelId || 'toan5ks1/Vintern-1B-v3_5-ONNX', data.device, data.dtype);
        } else if (type === 'process') {
            await processImage(data.image, data);
        }
    } catch (error) {
        self.postMessage({ status: 'error', error: error.message + '\n' + error.stack });
    }
});

async function loadModel(modelId, device = 'webgpu', dtypeConfig = null) {
    self.postMessage({ status: 'loading', message: `Loading model ${modelId} on ${device}...` });

    try {
        // Load tokenizer
        tokenizer = await AutoTokenizer.from_pretrained(modelId);
        
        // Configure dtype based on device or user config
        let dtype;
        if (dtypeConfig) {
            dtype = dtypeConfig;
        } else if (device === 'webgpu') {
            dtype = {
                embed_tokens: 'fp32',
                vision_encoder: 'fp32',
                decoder_model_merged: 'q4',
            };
        } else {
            // CPU configuration
            dtype = {
                embed_tokens: 'fp16',
                vision_encoder: 'fp16',
                decoder_model_merged: 'q4',
            };
        }

        // Load model
        model = await InternVLChatModel.from_pretrained(modelId, {
            dtype,
            device,
        });

        // Try to load processor
        try {
            processor = await AutoProcessor.from_pretrained(modelId);
        } catch (e) {
            console.log('AutoProcessor not available, will use manual preprocessing');
        }

        self.postMessage({ status: 'ready', message: 'Model loaded successfully!' });
    } catch (e) {
        console.error('Full error during model loading:', e);
        const errorMessage = e instanceof Error ? e.message : String(e);
        const errorStack = e instanceof Error ? e.stack : '';
        throw new Error(`Failed to load model: ${errorMessage}\nStack: ${errorStack}`);
    }
}

async function processImage(imageUrl, data = {}) {
    if (!model || !tokenizer) {
        throw new Error('Model not loaded');
    }

    self.postMessage({ status: 'processing', message: 'Processing image...' });

    // Load image
    const image = await RawImage.fromURL(imageUrl);

    // Preprocess image
    let pixel_values;
    if (processor) {
        const vision_inputs = await processor(image);
        pixel_values = vision_inputs.pixel_values;
    } else {
        pixel_values = await manualPreprocess(image);
    }

    // Create prompt
    const question = data.prompt || "Trích xuất thông tin và trả về dạng JSON.";
    const messages = [
        { role: 'user', content: `<image>\n${question}` }
    ];

    // Apply chat template
    let prompt = tokenizer.apply_chat_template(messages, { 
        tokenize: false, 
        add_generation_prompt: true 
    });

    // Replace <image> with 256 <IMG_CONTEXT> tokens
    // This matches the logic in vintern-test.js which worked correctly
    const image_tokens = "<IMG_CONTEXT>".repeat(256);
    prompt = prompt.replace("<image>", image_tokens);

    // Tokenize
    const inputs = tokenizer(prompt);
    inputs.pixel_values = pixel_values;

    // Create streamer
    const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
            self.postMessage({ status: 'update', output: text });
        }
    });

    self.postMessage({ status: 'generating', message: 'Generating response...' });

    // Generate
    let outputs;
    try {
        outputs = await model.generate({
            ...inputs,
            max_new_tokens: 1024,
            // repetition_penalty: 1.2,
            // do_sample: true,
            temperature: 0.75,
            top_p: 0.95,
            streamer,
        });
    } catch (e) {
        console.error('Generation failed:', e);
        throw new Error(`Generation failed: ${e.message}`);
    }

    // Decode
    const decoded = tokenizer.batch_decode(outputs, { skip_special_tokens: true });
    
    // Extract just the assistant's response
    const fullOutput = decoded[0];
    const assistantStart = fullOutput.lastIndexOf('assistant\n');
    const extractedText = assistantStart !== -1 
        ? fullOutput.substring(assistantStart + 'assistant\n'.length).trim()
        : fullOutput;

    self.postMessage({ status: 'complete', result: extractedText });
}

async function manualPreprocess(image) {
    // Resize to 448x448
    const resized = await image.resize(448, 448);
    
    // Normalization constants for InternVL
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    const data = resized.data;
    const width = resized.width;
    const height = resized.height;
    const channels = resized.channels;
    
    // Create float32 tensor
    const tensor = new Tensor('float32', new Float32Array(1 * 3 * 448 * 448), [1, 3, 448, 448]);
    
    for (let i = 0; i < width * height; ++i) {
        const offset = channels * i;
        const r = data[offset] / 255.0;
        const g = data[offset + 1] / 255.0;
        const b = data[offset + 2] / 255.0;
        
        tensor.data[i] = (r - mean[0]) / std[0]; // R channel
        tensor.data[width * height + i] = (g - mean[1]) / std[1]; // G channel
        tensor.data[2 * width * height + i] = (b - mean[2]) / std[2]; // B channel
    }
    
    return tensor;
}
