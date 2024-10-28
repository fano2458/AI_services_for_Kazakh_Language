import onnxruntime as ort
from kaz_img_caption.utils.language_utils import tokens2description, preprocess_image, create_pad_mask, create_no_peak_and_pad_mask
import pickle
import numpy as np
import shutil
from PIL import Image


class ImageCaptioningModel():
    def __init__(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        self.ort_session = ort.InferenceSession("/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/weights/checkpoints/kaz_model.onnx",
                                   sess_options=session_options, providers=providers)

        with open("/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/weights/checkpoints/vocabulary/vocab_kz.pickle", 'rb') as f:
                self.coco_tokens = pickle.load(f)
                self.sos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['sos_str']]
                self.eos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['eos_str']]

        batch_size = 1
        enc_exp_list = [32, 64, 128, 256, 512]
        dec_exp = 16
        num_heads = 8
        NUM_FEATURES = 144
        MAX_DECODE_STEPS = 20

        self.enc_mask = create_pad_mask(mask_size=(batch_size, sum(enc_exp_list), NUM_FEATURES),
                                    pad_row=[0], pad_column=[0]).contiguous()

        no_peak_mask = create_no_peak_and_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS),
                                                    num_pads=[0]).contiguous()

        cross_mask = create_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, NUM_FEATURES),
                                    pad_row=[0], pad_column=[0]).contiguous()
        # contrary to the other masks, we put 1 in correspondence to the values to be masked
        cross_mask = 1 - cross_mask

        self.fw_dec_mask = no_peak_mask.unsqueeze(2).expand(batch_size, MAX_DECODE_STEPS,
                                                        dec_exp, MAX_DECODE_STEPS).contiguous(). \
                            view(batch_size, MAX_DECODE_STEPS * dec_exp, MAX_DECODE_STEPS)

        self.bw_dec_mask = no_peak_mask.unsqueeze(-1).expand(batch_size,
                    MAX_DECODE_STEPS, MAX_DECODE_STEPS, dec_exp).contiguous(). \
                    view(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS * dec_exp)

        self.atten_mask = cross_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)


    def predict(self, path):
        file_path = f"kaz_img_caption/files/{path.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(path.file, buffer)
        
        pil_image = preprocess_image(path.file, 384)
        
        input_dict_1 = {'enc_x': pil_image.numpy(), 'sos_idx': np.array([self.sos_idx]),
                        'enc_mask': self.enc_mask.numpy(), 'fw_dec_mask': self.fw_dec_mask.numpy(),
                        'bw_dec_mask': self.bw_dec_mask.numpy(), 'cross_mask': self.atten_mask.numpy()}

        outputs_ort = self.ort_session.run(None, input_dict_1)
        output_caption = tokens2description(outputs_ort[0][0].tolist(), self.coco_tokens['idx2word_list'], self.sos_idx, self.eos_idx)
        
        return output_caption


if __name__ == "__main__":
    img_size = 384
    img_path = "/home/fano/Desktop/senior project/AI_services_for_Kazakh_Language/kaz_img_caption/files/image8.jpg"
    image_1 = preprocess_image(img_path, img_size)

    model = ImageCaptioningModel()
    model.predict(image_1)
