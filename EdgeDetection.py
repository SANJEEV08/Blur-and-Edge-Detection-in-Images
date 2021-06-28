import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
import streamlit as st 
from PIL import Image


# Apply automatic Canny edge detection using the computed median 
def edge_detect(gray_image) :
    
    # Sigma is parameter of gaussian blur
    sigma=0.33
    
    # Formula to calculate lower threshold value
    lower = int(max(0, (1.0 - sigma) * v))
    
    # Formula to calculate higher threshold value
    upper = int(min(255, (1.0 + sigma) * v))
    
    # To calculate edge detection following steps are performed:
    # Noise reduction
    # Gradient calculation
    # Non-maximum suppression
    # Double threshold
    # Edge Tracking by Hysteresis
    edged = cv2.Canny(gray_image, lower, upper)
    
    # To plot Grayscale image
    plt.subplot(121)
    plt.imshow(gray_image,cmap = 'gray')
    plt.title('Grayscale Image') 
    plt.xticks([])
    plt.yticks([])
    
    # To plot edge image
    plt.subplot(122)
    plt.imshow(edged,cmap = 'gray')
    plt.title('Edge Image')
    plt.xticks([]) 
    plt.yticks([])
    plt.show()

    
# Reading in an Image   
path = r'E:\Sanjeev\Projects-Python\Signal Processing\Final SP Project\Pollenflower.jpg'
orig = cv2.imread(path)

# Resizing the input image
image = imutils.resize(orig, width=500)

# Before any task performing we have to convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# We are finding median value of grayscale so that using median we are able to calculate threshold values
v = np.median(gray_image)

# Function call
edge_detect(gray_image)


# Topic - Web Application
 
st.title("😀 Welcome to Prediction Dashboard 😀")

html_temp = """
<div style="background-color:SteelBlue;padding:10px">
<h2 style="color:Wheat;text-align:center;">Edge Detection </h2>
</div>
"""

st.markdown(html_temp,unsafe_allow_html=True)

st.header(':wave: Hi Signal Processing Enthusiast !')

st.header('You just hopped on our :satellite_antenna: !')

st.subheader('Overview : This Web Application traces out the edges and the distribution of pollen grains in flowers which is needed to attain a balanced plantation :bulb:')

image = Image.open(path)

st.image(image, caption='Raw Uploaded Image', use_column_width=True)


if st.button("Diagnose"):
     result = Image.open('Pollenflowerresult.png')
     st.image(result, caption='Grayscale & Edge Results', use_column_width=True)
     
     
if st.button("About"):
    st.markdown(' Scripted with :yellow_heart: by Sanjeev',unsafe_allow_html=True)
    

st.subheader('Thanks for your time , Appreciate it :palms_up_together: !')



