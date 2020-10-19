import streamlit as st
import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

@st.cache
def load_image(image):
	img = Image.open(image)
	return img



def main():
	st.title("Face Detection App")
	st.text("Built with Streamlit and OpenCV")

	activities = ['Detection','About']
	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == 'Detection':
		st.subheader("Face Detection")
		
		image_file = st.file_uploader("Upload Image",type = ['jpg','png','jpeg'])

		if image_file is not None:
			our_img = load_image(image_file)
			st.subheader('Uploaded Image')
			st.image(our_img, width = 800)

	elif choice =='About':
		st.subheader("About Face Detection App")
		st.markdown("Built with Streamlit and OpenCV by [Rehan uddin](https://hardly-human.github.io/)")
		st.success("Rehan uddin (Hardly-Human)👋😉")
		














if __name__ == '__main__':
	main()