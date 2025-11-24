import React, { useState, useEffect, useRef } from 'react';

const MODEL_ID = 'toan5ks1/Vintern-1B-v3_5-ONNX';

const ResultViewer = ({ data, isProcessing }) => {
  const [parsedData, setParsedData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!data) return;
    
    // Don't attempt to parse while processing/streaming
    if (isProcessing) {
        setParsedData(null);
        return;
    }
    
    const parseJSON = (str) => {
      try {
        return JSON.parse(str);
      } catch (e) {
        return null;
      }
    };

    const extractAndParse = (text) => {
      // 1. Try cleaning markdown
      let clean = text.replace(/```json\n?|\n?```/g, '').trim();
      
      // 2. Try direct parse
      let parsed = parseJSON(clean);
      if (parsed) return parsed;

      // 3. Try to find the JSON object boundaries
      const firstOpen = clean.indexOf('{');
      const lastClose = clean.lastIndexOf('}');
      
      if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
        // Try the substring from first { to last }
        let candidate = clean.substring(firstOpen, lastClose + 1);
        parsed = parseJSON(candidate);
        if (parsed) return parsed;
        
        // 4. If that failed, maybe there are extra closing braces?
        // Try removing trailing characters until it parses or we run out
        let currentEnd = lastClose;
        while (currentEnd > firstOpen) {
            candidate = clean.substring(firstOpen, currentEnd + 1);
            parsed = parseJSON(candidate);
            if (parsed) return parsed;
            
            // Find the previous '}'
            currentEnd = clean.lastIndexOf('}', currentEnd - 1);
        }
      }
      
      throw new Error("Could not extract valid JSON");
    };

    try {
      const parsed = extractAndParse(data);
      setParsedData(parsed);
      setError(null);
    } catch (e) {
      console.warn("Failed to parse JSON:", e);
      setParsedData(null);
      setError(e);
    }
  }, [data, isProcessing]);

  const handleCopy = () => {
    navigator.clipboard.writeText(data);
  };

  if (isProcessing || error || !parsedData) {
    return (
      <div className="raw-result">
        <div className="result-header">
          <span className="label">{isProcessing ? 'Generating...' : 'Raw Output'}</span>
          <button className="copy-btn" onClick={handleCopy}>Copy</button>
        </div>
        <pre>{data}</pre>
      </div>
    );
  }

  const renderValue = (value) => {
    if (typeof value === 'object' && value !== null) {
      return (
        <div className="nested-object">
          {Object.entries(value).map(([k, v]) => (
            <div key={k} className="nested-row">
              <span className="nested-key">{k}:</span>
              <span className="nested-value">{renderValue(v)}</span>
            </div>
          ))}
        </div>
      );
    }
    return <span>{String(value)}</span>;
  };

  return (
    <div className="structured-result">
      <div className="result-header">
        <span className="label">Structured Data</span>
        <button className="copy-btn" onClick={handleCopy}>Copy JSON</button>
      </div>
      <div className="result-table">
        {Object.entries(parsedData).map(([key, value]) => (
          <div key={key} className="result-row">
            <div className="result-key">{key}</div>
            <div className="result-value">{renderValue(value)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

function App() {
  const [status, setStatus] = useState('Initializing...');
  const [statusType, setStatusType] = useState('loading');
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [device, setDevice] = useState('webgpu');
  const [embedDtype, setEmbedDtype] = useState('fp32');
  const [visionDtype, setVisionDtype] = useState('fp32');
  const [decoderDtype, setDecoderDtype] = useState('q4');
  const [prompt, setPrompt] = useState('Trích xuất thông tin và trả về dạng JSON.');
  
  const workerRef = useRef(null);

  const SUGGESTIONS = [
    { label: 'Extract JSON', value: 'Trích xuất thông tin và trả về dạng JSON.' },
    { label: 'Describe Image', value: 'Mô tả chi tiết hình ảnh này.' },
    { label: 'Read Text', value: 'Đọc tất cả nội dung văn bản trong ảnh.' },
  ];

  useEffect(() => {
    // Initialize worker
    workerRef.current = new Worker(new URL('./worker.js', import.meta.url), {
      type: 'module'
    });

    workerRef.current.onmessage = (e) => {
      const { status, message, result, error } = e.data;
      
      if (status === 'error') {
        setStatus(`❌ Error: ${error}`);
        setStatusType('error');
        setIsProcessing(false);
      } else if (status === 'ready') {
        setStatus(`✅ Model loaded successfully! (${device.toUpperCase()})`);
        setStatusType('success');
        setIsModelReady(true);
      } else if (status === 'update') {
        setResult(prev => (prev || '') + e.data.output);
        setStatus('Generating...');
      } else if (status === 'complete') {
        setStatus('✅ Processing complete!');
        setStatusType('success');
        // We might not need to set result here if streaming worked, 
        // but let's do it to be safe and ensure clean final state
        setResult(result);
        setIsProcessing(false);
      } else {
        setStatus(message);
        setStatusType('loading');
      }
    };

    // Load model
    setIsModelReady(false);
    setStatus(`Loading model on ${device.toUpperCase()}...`);
    workerRef.current.postMessage({
      type: 'load',
      data: { 
        modelId: MODEL_ID,
        device: device,
        dtype: {
            embed_tokens: embedDtype,
            vision_encoder: visionDtype,
            decoder_model_merged: decoderDtype
        }
      }
    });

    return () => {
      workerRef.current?.terminate();
    };
  }, [device, embedDtype, visionDtype, decoderDtype]);

  // Auto-load default image
  useEffect(() => {
    setImage('/images/xx.jpg');
  }, []);



  const handleFileUpload = (file) => {
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target.result);
      setResult(null);
    };
    reader.readAsDataURL(file);
  };

  const handleProcess = () => {
    if (!image || !isModelReady) return;
    
    setIsProcessing(true);
    setResult(null);
    
    workerRef.current.postMessage({
      type: 'process',
      data: { 
        image, 
        prompt
      }
    });
  };

  return (
    <div className="container">
      <h1>🚀 Vintern WebGPU App</h1>
      <p className="subtitle">Vietnamese OCR Model with GPU Acceleration</p>
      
      <div className="main-content">
        <div className="input-section">
          <div 
            className={`upload-area ${image ? 'has-image' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              e.currentTarget.classList.add('dragover');
            }}
            onDragLeave={(e) => {
              e.currentTarget.classList.remove('dragover');
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.currentTarget.classList.remove('dragover');
              if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
              }
            }}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <div className="upload-icon">📸</div>
            <p><strong>Click to upload</strong> or drag and drop an image</p>
            <p style={{ color: '#666', fontSize: '0.9em', marginTop: '5px' }}>PNG, JPG up to 10MB</p>
            <input 
              type="file" 
              id="fileInput" 
              accept="image/*" 
              onChange={(e) => handleFileUpload(e.target.files[0])}
            />
          </div>

          {image && (
            <div className="preview-container">
              <img id="imagePreview" src={image} alt="Preview" />
            </div>
          )}
        </div>

        <div className="output-section">
          <div className="controls-card">
            <div className="controls" style={{ marginBottom: '20px' }}>
              <label style={{ marginRight: '10px', display: 'block', marginBottom: '5px', fontWeight: '500' }}>Processing Device:</label>
              <select 
                value={device} 
                onChange={(e) => setDevice(e.target.value)}
                disabled={isProcessing}
                style={{ padding: '8px', borderRadius: '6px', width: '100%', border: '1px solid #ddd', marginBottom: '10px' }}
              >
                <option value="webgpu">WebGPU (Fast, Low Memory)</option>
                <option value="wasm">CPU (Slow, High Memory)</option>
              </select>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                  <div>
                      <label style={{ fontSize: '0.8em', fontWeight: '500', display: 'block', marginBottom: '3px' }}>Embed Tokens</label>
                      <select 
                        value={embedDtype} 
                        onChange={(e) => setEmbedDtype(e.target.value)}
                        disabled={isProcessing}
                        style={{ padding: '5px', borderRadius: '4px', width: '100%', border: '1px solid #ddd' }}
                      >
                        <option value="fp32">fp32</option>
                        <option value="fp16">fp16</option>
                      </select>
                  </div>
                  <div>
                      <label style={{ fontSize: '0.8em', fontWeight: '500', display: 'block', marginBottom: '3px' }}>Vision Encoder</label>
                      <select 
                        value={visionDtype} 
                        onChange={(e) => setVisionDtype(e.target.value)}
                        disabled={isProcessing}
                        style={{ padding: '5px', borderRadius: '4px', width: '100%', border: '1px solid #ddd' }}
                      >
                        <option value="fp32">fp32</option>
                        <option value="fp16">fp16</option>
                        <option value="q8">q8</option>
                      </select>
                  </div>
                  <div>
                      <label style={{ fontSize: '0.8em', fontWeight: '500', display: 'block', marginBottom: '3px' }}>Decoder</label>
                      <select 
                        value={decoderDtype} 
                        onChange={(e) => setDecoderDtype(e.target.value)}
                        disabled={isProcessing}
                        style={{ padding: '5px', borderRadius: '4px', width: '100%', border: '1px solid #ddd' }}
                      >
                        <option value="fp32">fp32</option>
                        <option value="fp16">fp16</option>
                        <option value="q8">q8</option>
                        <option value="q4">q4</option>
                        <option value="q4f16">q4f16</option>
                      </select>
                  </div>
              </div>
            </div>

            <div className="prompt-section" style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: '500' }}>Prompt:</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isProcessing}
                className="prompt-input"
                rows="3"
              />
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button 
                    key={s.label} 
                    className="suggestion-chip"
                    onClick={() => setPrompt(s.value)}
                    disabled={isProcessing}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            <button 
              onClick={handleProcess} 
              disabled={!image || !isModelReady || isProcessing}
            >
              {isProcessing ? 'Processing...' : 'Process Image'}
            </button>

            {status && (
              <div id="status" className={statusType}>
                {statusType === 'loading' && <div className="spinner"></div>}
                {status}
              </div>
            )}
          </div>

          {result && (
            <div className="result-container">
              <h3>📄 Extracted Information</h3>
              <ResultViewer data={result} isProcessing={isProcessing} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
