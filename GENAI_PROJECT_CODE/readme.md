# AI Conversational System

An intelligent conversational system that combines Retrieval-Augmented Generation (RAG), voice cloning, and video synthesis to create interactive AI-powered conversations with realistic voice and video output.

## Features

- **Document-based Q&A**: Query PDF documents using RAG (Retrieval-Augmented Generation)
- **Voice Cloning**: Generate speech in a cloned voice using OpenVoice
- **Video Synthesis**: Create lip-synced videos using LatentSync
- **Interactive Interface**: Command-line interface for real-time queries

## Architecture

The system consists of three main components:

1. **RAG Application** (`rag_script.py`): Processes documents and answers questions
2. **Voice Clone** (`OpenVoice/voice_clone_script.py`): Clones and synthesizes speech
3. **Video Generation** (`LatentSync/video_gen.py`): Creates lip-synced videos

## Prerequisites

- Python 3.10.13
- Conda package manager
- Ubuntu/Linux (for apt dependencies)
- HuggingFace account and token
- OpenAI API key (for RAG functionality)

## Installation

### 1. Clone Required Repositories

```bash
# Clone LatentSync
git clone https://github.com/bytedance/LatentSync.git

# Clone OpenVoice
git clone https://github.com/myshell-ai/OpenVoice.git
```

### 2. Environment Setup

Run the setup script to create and configure the conda environment:

```bash
#!/bin/bash
# Create a new conda environment
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints
```

### 3. OpenVoice Setup

```bash
# Install MeloTTS for OpenVoice
pip install git+https://github.com/myshell-ai/MeloTTS.git

# Download unidic for Japanese language support
python -m unidic download
```

### 4. Configuration

1. **HuggingFace Token**: Add your HuggingFace token to the login function:
   ```python
   login(token="your_huggingface_token_here")
   ```

2. **OpenAI API Key**: Set your OpenAI API key:
   ```python
   key = "your_openai_api_key_here"
   ```

## Project Structure

```
project/
├── main.py                           # Main application script
├── rag_script.py                     # RAG implementation
├── documents/
│   └── diabetes.pdf                  # Sample document
├── OpenVoice/
│   ├── voice_clone_script.py         # Voice cloning implementation
│   ├── resources/
│   │   ├── sarvesh.mp3              # Reference speaker audio
│   │   └── intro_audio.wav          # Alternative reference audio
│   └── outputs_v2/                  # Generated audio outputs
├── LatentSync/
│   ├── video_gen.py                 # Video generation implementation
│   ├── configs/unet/stage2.yaml     # UNet configuration
│   ├── weights/latentsync_unet.pt   # Model weights
│   └── assets/
│       └── demo1_video.mp4          # Sample video for lip-sync
└── requirements.txt                  # Python dependencies
```

## Usage

### Basic Usage

1. **Activate the environment**:
   ```bash
   conda activate latentsync
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Enter your query** when prompted:
   ```
   Enter your query: What is type 1 diabetes?
   ```

The system will:
- Process your question using the RAG system
- Generate an answer based on the document
- Create audio output using voice cloning
- Optionally generate a lip-synced video

### Advanced Usage

To enable video generation, uncomment the video synthesis section in `main.py`:

```python
# Uncomment these lines for video generation
unet_config_path = "LatentSync/configs/unet/stage2.yaml"
inference_ckpt_path = "LatentSync/weights/latentsync_unet.pt"
infer_class = CustomLatentSync(unet_config_path, inference_ckpt_path)
print("model_loaded")

video_path = "LatentSync/assets/demo1_video.mp4"
audio_path = "OpenVoice/outputs_v2/output_v2_1.wav"
video_out_path = "video_out_male.mp4"
inference_steps = 6
guidance_scale = 1.0

infer_class.infer(video_path, audio_path, video_out_path, inference_steps, guidance_scale)
```

## Configuration Options

### Voice Cloning
- **Reference Audio**: Change the reference speaker by modifying `ref_spk_audio`
- **Output Quality**: Adjust voice cloning parameters in `VoiceClone` class

### Video Generation
- **Inference Steps**: Control generation quality (higher = better quality, slower)
- **Guidance Scale**: Adjust adherence to input video (1.0 recommended)
- **Input Video**: Replace `demo1_video.mp4` with your preferred base video

### RAG System
- **Documents**: Add new PDF documents to the `documents/` folder
- **Vector Store**: The system automatically creates embeddings for fast retrieval

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure CUDA is properly installed for GPU acceleration
2. **Audio Dependencies**: Install system audio libraries if audio processing fails
3. **Model Download**: Ensure HuggingFace CLI is authenticated for model downloads
4. **Memory Issues**: Reduce batch sizes or use CPU inference for limited memory systems

### Dependencies

If you encounter package conflicts, create a fresh environment:

```bash
conda deactivate
conda remove -n latentsync --all
# Then repeat installation steps
```

## Performance Notes

- **GPU Recommended**: Video generation and voice cloning benefit significantly from GPU acceleration
- **Memory Requirements**: Video generation requires substantial RAM (8GB+ recommended)
- **Processing Time**: Initial setup includes model downloads and may take several minutes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project combines multiple open-source components:
- LatentSync: ByteDance License
- OpenVoice: MyShell License
- Custom components: MIT License

## Acknowledgments

- [ByteDance LatentSync](https://github.com/bytedance/LatentSync) for video synthesis
- [MyShell OpenVoice](https://github.com/myshell-ai/OpenVoice) for voice cloning
- OpenAI for language model capabilities

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review component-specific documentation
3. Open an issue with detailed error logs