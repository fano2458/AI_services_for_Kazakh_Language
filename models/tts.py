from espnet_onnx.export import TTSModelExport

m = TTSModelExport()
m.export_from_zip(
  '/home/fano/Downloads/kaztts_male1_tacotron2_train.loss.ave.zip',
  tag_name='kaztts',
  quantize=False
)

from scipy.io.wavfile import write

from espnet_onnx import Text2Speech

fs = 22050

tag_name = 'kaztts'
text2speech = Text2Speech(tag_name)

text = 'бүгінде өңірде тағы бес жобаның құрылысы жүргізілуде.'
output_dict = text2speech(text)

wav = output_dict['wav']

# import IPython.display as ipd
# ipd.Audio(output_dict['wav'], rate=22050)

folder_to_save, wav_name = "/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/models/generated_wav", "example6.wav"

# Path(folder_to_save).mkdir(parents=True, exist_ok=True)
# print(type(wav))
write(folder_to_save + "/" + wav_name, fs, wav)#, wav.view(-1).cpu().numpy())