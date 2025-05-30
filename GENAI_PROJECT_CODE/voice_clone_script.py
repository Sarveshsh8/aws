import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS


class VoiceClone:
    def __init__(self, ref_speaker_audio: str):
        ckpt_converter = 'OpenVoice/checkpoints_v2/converter'

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dir = 'OpenVoice/outputs_v2'

        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        os.makedirs(self.output_dir, exist_ok=True)
        self.target_se, audio_name = se_extractor.get_se(ref_speaker_audio, self.tone_color_converter, vad=True)

        language = "EN"
        self.model = TTS(language=language, device=device)
        self.speaker_ids = self.model.hps.data.spk2id

        self.source_se = torch.load(f'OpenVoice/checkpoints_v2/base_speakers/ses/en-india.pth', map_location=device)

        self.speaker_key = "en-india"
        self.speaker_id = 2

        # self.speaker_key = "en-us"
        # self.speaker_id = 0

    def infer_voice_clone(self, text):
        src_path = f'{self.output_dir}/tmp.wav'
        self.model.tts_to_file(text, self.speaker_id, src_path, speed=1.0)
        save_path = f'{self.output_dir}/output_v2_1.wav'

        print("save-path",save_path)

        # Run the tone color converter
        encode_message = "@MyShell"
        self.tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=self.source_se, 
            tgt_se=self.target_se, 
            output_path=save_path,
            message=encode_message)