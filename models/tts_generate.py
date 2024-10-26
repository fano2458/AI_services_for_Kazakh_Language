import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
from scipy.io.wavfile import write
from pathlib import Path
from fastapi import Response


class TTS():
    """
    Text-To-Speech (TTS) class for Kazakh language.
    This class generates speech waveforms from a given text input. It utilizes a pre-trained Tacotron2 model
    for text processing and a parallel WaveGAN model for waveform generation.
    """
    def __init__(self):
        """
        Initializes the TTS model and vocoder.
        Loads the pre-trained Tacotron2 model, WaveGAN vocoder, and sets their configurations.
        """

        self.fs = 22050

        vocoder_checkpoint="/home/fano/Downloads/parallelwavegan_male1_checkpoint/checkpoint-400000steps.pkl"
        self.vocoder = load_model(vocoder_checkpoint).to("cuda").eval()
        self.vocoder.remove_weight_norm()

        config_file = "/home/fano/Downloads/kaztts_male1_tacotron2_train.loss.ave/exp/tts_train_raw_char/config.yaml"
        model_path = "/home/fano/Downloads/kaztts_male1_tacotron2_train.loss.ave/exp/tts_train_raw_char/train.loss.ave_5best.pth"

        self.text2speech = Text2Speech(
            config_file,
            model_path,
            device="cuda",
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=True,
            backward_window=1,
            forward_window=3,
        )

    def predict(self, sample_text):
        """
        Generates a speech waveform from the provided text.

        Args:
            sample_text (str): The text to be converted to speech.

        Returns:
            fastapi.Response: A FastAPI response object with the generated audio waveform as content.
        """
        
        with torch.no_grad():
            output_dict = self.text2speech(sample_text.lower())
            feat_gen = output_dict['feat_gen']
            wav = self.vocoder.inference(feat_gen)
            
        folder_to_save, wav_name = "/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/models/generated_wav", "example5.wav"

        Path(folder_to_save).mkdir(parents=True, exist_ok=True)
        write(folder_to_save + "/" + wav_name, self.fs, wav.view(-1).cpu().numpy())

        wav_data = wav.view(-1).cpu().numpy()
        file_path = "/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/models/generated_wav/example5.wav"

        with open(file_path, 'rb') as wav_file:
            wav_data = wav_file.read()

        response = Response(content=wav_data, media_type="audio/wav")
        response.headers["Content-Disposition"] = "attachment; filename=my_wav_file.wav"

        return response
