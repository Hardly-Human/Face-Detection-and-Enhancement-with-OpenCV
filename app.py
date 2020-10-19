import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# loads image
@st.cache  # caches the data for efficiency..
def load_image(image):
	img = Image.open(image)
	return img

# detects faces and draws rectangles around them..
@st.cache
def detect_faces(img):
	new_img = np.array(img.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY) 
	# Detect Face
	faces = face_cascade.detectMultiScale(gray,1.3,4)
	# Draw Rectangle
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)

	return img,faces

# detects eyes and draws rectangles around them..
@st.cache
def detect_eyes(img):
	new_img = np.array(img.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY) 
	# Detect Face
	eyes = eye_cascade.detectMultiScale(gray,1.3,5)
	# Draw Rectangle
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(img, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

# cartoonizes the Image..
@st.cache
def cartonize_image(img):
	new_img = np.array(img.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY) 
	# Edges 
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
	# Color
	color = cv2.bilateralFilter(img , 9 ,300,300)
	# Cartoon
	cartoon = cv2.bitwise_and(color,color,mask=edges)

	return cartoon

# changes the orientation of Image..
@st.cache
def cannize_image(img):
	new_img = np.array(img.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny


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

		img_task = st.sidebar.selectbox("Select Task",['Enhance Image','Find Features in Image'])	
		
		if img_task == 'Enhance Image':	

			enhance_type = st.sidebar.radio("Enhance Type",['None','Gray-Scale','Contrast','Brightness','Blurring'])

			if enhance_type == 'None':
				st.warning("Upload Image and Select a Task")

			elif enhance_type == 'Gray-Scale':
				new_img = np.array(our_img.convert('RGB'))
				img = cv2.cvtColor(new_img,1)
				gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				st.subheader('Processed Image')
				st.image(gray_img,width = 800)

			elif enhance_type == 'Contrast':
				c_rate = st.sidebar.slider("Contrast",0.5,3.5)
				enhancer = ImageEnhance.Contrast(our_img)
				img_output = enhancer.enhance(c_rate)
				st.subheader('Processed Image')
				st.image(img_output,width = 800)

			elif enhance_type == 'Brightness':
				c_rate = st.sidebar.slider("Brightness",0.5,3.5)
				enhancer = ImageEnhance.Brightness(our_img)
				img_output = enhancer.enhance(c_rate)
				st.subheader('Processed Image')
				st.image(img_output,width = 800)

			elif enhance_type == "Blurring":
				new_img = np.array(our_img.convert('RGB'))
				blur_rate = st.sidebar.slider("Blurring",0.5,3.5)
				img = cv2.cvtColor(new_img,1)
				blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
				st.subheader('Processed Image')
				st.image(blur_img,width = 800)
			
			else:
				st.image(our_img, width = 800)





	elif choice =='About':
		st.subheader("About Face Detection App")
		st.markdown("Built with Streamlit and OpenCV by [Rehan uddin](https://hardly-human.github.io/)")
		st.success("Rehan uddin (Hardly-Human)👋😉")
		














if __name__ == '__main__':
	main()