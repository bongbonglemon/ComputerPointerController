'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class EyesDetect:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, output, device="CPU", threshold=0.60):
        '''
        TODO: Use this to set your instance variables.
        '''
        
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.exec_net=None
        self.output=output

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        self.exec_net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        start_time = time.time()
        input_dict={self.input_name:processed_image}
        print("Time take for landmark detection inference (in seconds):", time.time()-start_time)
        outputs = self.exec_net.infer(input_dict)
        face = self.preprocess_output(outputs, image)
        return face

    def check_model(self):
        #KIV
        return

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        width, height = 48, 48
        image = cv2.resize(image, (width, height))
        image = image.transpose((2,0,1))
        image = image.reshape(1, 3, height, width)
        return image

    def get_eye(self, x, y, size, image):
        left = x - size
        top = y - size
        crop_img = image[top:top+size*2, left:left+size*2]

        return crop_img

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        height,width = image.shape[0], image.shape[1]
        # TODO: detections might be put under a key in output blob
        landmarks = outputs['95']
        l_eye_x = int(landmarks[0,0,0,0]*width)
        l_eye_y = int(landmarks[0,1,0,0]*height)
        r_eye_x = int(landmarks[0,2,0,0]*width)
        r_eye_y = int(landmarks[0,3,0,0]*height)
        size = 30

        # account for edge case: eye too close to side of crop (KIV)
        if l_eye_x < 30:
            size=20
        if l_eye_x < 20:
            size=15

        
        left_eye = self.get_eye(l_eye_x, l_eye_y, size, image)
        right_eye = self.get_eye(r_eye_x, r_eye_y, size, image)

        if self.output == True:
            cv2.imwrite("left_eye.png", left_eye)
            cv2.imwrite("right_eye.png", right_eye)
        return left_eye, right_eye

    
