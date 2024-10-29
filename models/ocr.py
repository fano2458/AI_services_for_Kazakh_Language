from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

import base64
from io import BytesIO


class OCRModel():
    def __init__(self):
        self.langs = ["kk", "ru", "en"] 

        self.det_processor, self.det_model = load_det_processor(), load_det_model()
        self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()


    def predict(self, img):
        img = base64.b64decode(img)
        image_stream = BytesIO(img)
        image = Image.open(image_stream)
        # image = Image.open(IMAGE_PATH)
        predictions = run_ocr([image], [self.langs], self.det_model, self.det_processor, self.rec_model, self.rec_processor)
        formatted_text = ""

        for result in predictions:
            text_lines = result.text_lines  # Extract text lines from OCRResult
            for line in text_lines:
                if line.confidence >= 0.50:  # Check confidence threshold
                    formatted_text += line.text + "\n"  # Append the text to the result

        return formatted_text