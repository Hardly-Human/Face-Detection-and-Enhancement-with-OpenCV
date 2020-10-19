import streamlit as st
import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')


def main():
	st.title("Face Detection App")
	st.text("Built with Streamlit and OpenCV")


if __name__ == '__main__':
	main()