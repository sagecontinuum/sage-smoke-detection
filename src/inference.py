import abc
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import sys

class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'inference') or NotImplemented)

    @abc.abstractmethod
    def inference(self,current_image,previous_image,smoke_threshold=0.5):
        """Perform model inference"""
        raise NotImplementedError

class ModelBase(ModelInterface):
    def __init__(self,modelPath):
        self.modelPath = modelPath

class SmokeyNet(ModelBase):
    def __init__(self, modelPath):
        super().__init__(modelPath)
        self.check_model()

    def check_model(self):
        onnx_model = onnx.load(self.modelPath)
        result = onnx.checker.check_model(onnx_model)
        if result is not None:
            sys.exit('Onnx model failed checker')

    def start_session(self):
        return onnxruntime.InferenceSession(self.modelPath)

    def run_ort(self,ort_session, next_image,current_image,smoke_threshold):
        x = self.generate_input_data(next_image,current_image)
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs =  ort_outs[0]
        tile_preds, tile_probs = self.get_preds_and_probs(outputs,smoke_threshold)
        image_preds = (tile_preds.sum(axis=1) > 0)
        return image_preds, tile_preds, tile_probs

    def inference(self,next_image,current_image,smoke_threshold=0.5):
        ort_session = self.start_session()
        image_preds, tile_preds, tile_probs = self.run_ort(ort_session, next_image,current_image,smoke_threshold)
        return image_preds, tile_preds, tile_probs

    def sigmoid(self,z):
        return 1.0/(1.0 + np.exp(-z))

    def normalize_image(self,img):
        img = img / 255
        img = (img - 0.5) / 0.5
        return img

    def calculate_num_tiles(self,resize_dimensions, crop_height, tile_dimensions, tile_overlap):
        num_tiles_height = 1 + (crop_height - tile_dimensions[0]) // (tile_dimensions[0] - tile_overlap)
        num_tiles_width = 1 + (resize_dimensions[1] - tile_dimensions[1]) // (tile_dimensions[1] - tile_overlap)
        return num_tiles_height, num_tiles_width

    def tile_image(self,img, num_tiles_height, num_tiles_width, resize_dimensions,
                        tile_dimensions, tile_overlap):
        bytelength = img.nbytes // img.size
        img = np.lib.stride_tricks.as_strided(img,
            shape=(num_tiles_height,
                num_tiles_width,
                tile_dimensions[0],
                tile_dimensions[1],
                3),
            strides=(resize_dimensions[1]*(tile_dimensions[0]-tile_overlap)*bytelength*3,
                    (tile_dimensions[1]-tile_overlap)*bytelength*3,
                    resize_dimensions[1]*bytelength*3,
                    bytelength*3,
                    bytelength), writeable=False)
        img = img.reshape((-1, tile_dimensions[0], tile_dimensions[1], 3))
        return img

    def preprocess_image(self,img_array,num_tiles_height,num_tiles_width,resize_dimensions,
                            crop_height,tile_dimensions,tile_overlap):
        img = Image.fromarray(img_array)
        img = img.resize((resize_dimensions[1],resize_dimensions[0]))
        img = np.array(img)[-crop_height:]
        img = self.tile_image(img, num_tiles_height,
                                num_tiles_width, resize_dimensions,
                                tile_dimensions, tile_overlap)
        img = self.normalize_image(img)
        return img

    def generate_input_data(self,curr_img,prev_img):
        resize_dimensions = (1392, 1856)
        crop_height = 1040
        tile_dimensions = (224,224)
        tile_overlap = 20
        num_tiles_height, num_tiles_width = self.calculate_num_tiles(resize_dimensions,
                                                                        crop_height,
                                                                        tile_dimensions,
                                                                        tile_overlap,
                                                                    )
        imgs = [curr_img,prev_img]
        x = []
        for img in imgs:
            img = self.preprocess_image(img,
                                        num_tiles_height,
                                        num_tiles_width,
                                        resize_dimensions,
                                        crop_height,
                                        tile_dimensions,
                                        tile_overlap,
                                        )
            x.append(img)
        x = np.transpose(np.stack(x), (1, 0, 4, 2, 3))
        x = np.expand_dims(x, axis=0)
        return x

    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def get_preds_and_probs(self,tile_outputs,smoke_threshold=0.5):
        tile_probs = self.sigmoid(tile_outputs)
        tile_preds = (tile_probs > smoke_threshold).astype("uint8")
        return tile_preds, tile_probs