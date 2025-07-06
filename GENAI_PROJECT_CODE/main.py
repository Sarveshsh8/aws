# from rag_script import RAGApplication
# from OpenVoice.voice_clone_script import VoiceClone

import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")


# Optionally suppress all warnings (not recommended for production)
# warnings.filterwarnings("ignore")

import sys
sys.path.append("/OpenVoice")

from rag_script import RAGApplication
from OpenVoice.voice_clone_script import VoiceClone
from LatentSync.video_gen import CustomLatentSync
from huggingface_hub import login

login(token="")


if __name__ == "__main__":
#     # Initialize
        key = ""
        rag = RAGApplication(key)

        # Setup document (first time)
        rag.setup_document("documents/diabetes.pdf")

        # Ask questions
        # result = rag.ask_question("What is a type1 diabetes?")
        query = input("Enter your query:   ")
        result = rag.ask_question(query)
        print(result['answer'])


        # ref_spk_audio = "OpenVoice/resources/sarvesh.mp3"
        ref_spk_audio = "OpenVoice/resources/intro_audio.wav"
        cloner = VoiceClone(ref_spk_audio)
        
        cloner.infer_voice_clone(result['answer'])


        # unet_config_path = "LatentSync/configs/unet/stage2.yaml"
        # inference_ckpt_path = "LatentSync/weights/latentsync_unet.pt"

        # infer_class = CustomLatentSync(unet_config_path, inference_ckpt_path)
        # print("model_loaded")
        # video_path = "LatentSync/assets/demo1_video.mp4"
        # audio_path = "OpenVoice/outputs_v2/output_v2_1.wav"
        # video_out_path = "video_out_male.mp4"
        # inference_steps = 6
        # guidance_scale = 1.0
        # infer_class.infer(video_path, audio_path, video_out_path, inference_steps, guidance_scale)