#!/usr/bin/env python

import roslib
roslib.load_manifest('video_stream_python')
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import model_from_json, load_model
from random import randint
from detection import detect
import time

from steer import SegmentToSteer
import configparser

config = configparser.RawConfigParser()
# configFilePath = r'/home/hieung1707/catkin_ws/src/video_stream_python/scripts/config.env1'
config.read(os.path.join(os.path.dirname(__file__), 'config.env1'))
end = time.time()

class processor:
	def __init__(self):
		self.image = None
		self.model = self.load_model_segment()
		self.graph = tf.get_default_graph()
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/Team1_image/compressed', CompressedImage, self.callback, queue_size=1)
		self.pub_speed = rospy.Publisher('/Team1_speed', Float32, queue_size=1)
		self.pub_steerAngle = rospy.Publisher('/Team1_steerAngle', Float32, queue_size=1)
		self.lastTime = time.time()
		self.s2s = SegmentToSteer(square=3, margin=10, roi=0.5)

	def load_model_segment(self):
		json_file = open(config.get('my-config', 'MODEL_SEGMENT_GRAPH_PATH'), 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		m = model_from_json(loaded_model_json)
		m.load_weights(config.get('my-config', 'MODEL_SEGMENT_WEIGHT_PATH'))
		m.predict(np.zeros((1, 160, 320, 3), dtype=np.float32))
		print("Predicted")
		print("Model Loaded")
		return m

	def callback(self, data):
		global end
		if time.time() - end >= 0.005:
			try:
				with self.graph.as_default():
					self.image = self.convert_data_to_image(data.data)
					img_cpy = self.image.copy()
					flag, s = detect(img_cpy)
					cv2.imshow('image', img_cpy)
					cv2.waitKey(1)
					y = rospy.get_time()
					# if s != 0:
					# 	print(flag, s)
					res = self.get_segment_image(self.image)
					steer, res = self.s2s.get_steer(res*255., flag)
					# speed = 60*np.cos(abs(steer)*np.pi/180)
					speed = 60
					cv2.imshow('segment', res*1.)
					cv2.waitKey(1)
					self.publish_data(speed, steer)

			except CvBridgeError as e:
				print(e)

		end = time.time()

	def convert_data_to_image(self, data):
		arr = np.fromstring(data, np.uint8)
		image = cv2.imdecode(arr, 1)
		#240, 320
		return cv2.resize(image, (320,160))

	def get_segment_image(self, image):
		res = self.model.predict(np.expand_dims(image/255., axis=0))
		return np.argmax(res, axis=3)[0]

	def publish_data(self, speed, steerAngle):
		self.pub_speed.publish(speed)
		self.pub_steerAngle.publish(steerAngle)	

def main(args):
	p = processor()
	rospy.init_node('video_stream_python')
	try:
		rospy.spin()
	except:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)