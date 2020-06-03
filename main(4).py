"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import numpy as np
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
def draw_boxes(frame, result):
    
    '''
    Draw bounding boxes onto the frame.
    
    '''
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        #gievn conf greater than 0.5,because in build parser function  prob_threshold set default as 0.5
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            count +=1
    
            
    return frame,count

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    return parser

def capture(args):
    single_image_mode = False
    if args == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.endswith('.jpg') or args.endswith('.bmp') :
        single_image_mode = True
        input_stream = args

    # Checks for video file
    else:
        input_stream = args
    return input_stream,single_image_mode
def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    #ading requring variable
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    #arguments like args.m, .d,.cpu_extension, in bulid_argparsar function
    infer_network.load_model(args.model,args.device,args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###

    ### TODO: Loop until stream is over ###
    stream_input, mode = capture(args.input)
    single_image_mode = mode
    cap = cv2.VideoCapture(stream_input)
    cap.open(args.input)
    #width and height is import parameters in pretrained model
    global width,height
    width = int(cap.get(3))
    height = int(cap.get(4))
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # Read the next frame
        #in this captured input stay in frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Read from the video capture ###

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame,(net_input_shape[3],net_input_shape[2] ))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
         # https://github.com/intel-iot-devkit/people-counter-python/blob/master/main.py taken refernce for this section
        inf_start = time.time()
        infer_network.exec_net(p_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            # Results of the output layer of the network
            result = infer_network.get_output(cur_request_id)
            if args.perf_counts:
                perf_count = infer_network.performance_counter(cur_request_id)
                performance_counts(perf_count)
### TODO: Get the results of the inference request ###
            result, count = draw_boxes(p_frame, result)
            #Display inference time
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(result, inf_time_message, (15, 15),
                       cv2.FONT_HERSHEY_COMPLEX, 0.45, (200, 10, 10), 1)
            #time = time.time()

            ### TODO: Extract any desired stats from the results ###
            #client.publish("time",json.dumps({"Time":time}))
            if count > last_count:
                start_time = time.time()
                total_count = total_count + count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            if count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": count}))
            last_count = count

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
       # output = np.ascontiguousarray(output, dtype=np.float32)

        ### TODO: Send the frame to the FFMPEG server ###
        #output = cv2.resize(output,(net_input_shape[3],net_input_shape[2]))
       # frame = np.dstack((result,result,result))
        #frame = np.uint8(result)
        sys.stdout.buffer.write(result)
        sys.stdout.flush()
        #print(output)

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
                cv2.imwrite('output_image.jpg', result)


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
