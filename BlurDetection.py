# Importing Necessary Packages

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
from PIL import Image
import imutils
import cv2

# Function Parameters definition 
# image - Input image for Blur Detection
# size - Represents the area of concentration for FFT to be applied
# thresh - Threshold value with which the image will be graded as bluury / Not blurry
# vis - A boolean indicating whether to visualize/plot the original input image and magnitude image using matplotlib

def detect_blur_fft(image, size=60, thresh=10, vis=1):
    
	# Getting the dimension of the input image that is the height & width
    (h, w) = image.shape
    # Getting the center corrdinates as height/2 & width/2 and int() is used to approximate the values
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # Compute the FFT to find the Frequency Transform representation
    fft = np.fft.fft2(image)
	# Shifting the zero frequency component (i.e., DC component located at the top-left corner) to the center where it will be more easy to analyze
    # In other words rearranging the values so that all the values will be either to left/right of zero
    fftShift = np.fft.fftshift(fft)
    
	
    # check to see if we are visualizing our output
    if vis==1:
		# Computing the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fftShift))
		# Displaying the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Grayscale Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # Displaying the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # Plotting original input image and magnitude image side by side
        plt.show()
		
		
		
		
	# Topic - Image Reconstruction
    
    # Removing low frequency components (High pass filter)
    # Image smoothing (Edges & lines are not observed) - A blurry image is formed out of a low pass filter .
    # Image sharpening (Edges & lines are enhanced) - This occurs at high frequency
    # At the end, When Magnitude > threshold it is a sharp image (Not blurry) & When Magnitude < threshold is not a sharp image (blurry)
    # Here we are comparing the degree of sharpness to determine whether an image is blurry or not
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    # Rearrange the values with 0th value at left top
    fftShift = np.fft.ifftshift(fftShift)
	# IFFT to construct a new modified image
    recon = np.fft.ifft2(fftShift)
	
    
    
    # Topic - Blur Checking
    
    # Computing the magnitude spectrum of the reconstructed image
    magnitude = 20 * np.log(np.abs(recon))
    # Computing the mean of the reconstructed image
    mean = np.mean(magnitude)
	# If mean <= threshold it is considered blurry 
    return (mean, mean <= thresh)




# Topic - Initial preparation & Inputs

# Reading in an Image
path = r'E:\Sanjeev\Projects-Python\Signal Processing\Final SP Project\NotBlur1.jpg'
orig = cv2.imread(path)
# Resizing the input image
orig = imutils.resize(orig, width=500)
# Before any task performing we have to convert the image to grayscale
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
# Function call for blur detecton
(mean, blurry) = detect_blur_fft(gray, size=60,thresh=10, vis=1)


# Topic - Printing Results

print(f"Mean of Magnitude Spectrum :{mean}")
print(f"Blurry Image ? :{blurry}")



# Topic - Web Application
 
st.title("😀 Welcome to Prediction Dashboard 😀")

html_temp = """
<div style="background-color:SteelBlue;padding:10px">
<h2 style="color:Wheat;text-align:center;">Blurry Image Diagnostics </h2>
</div>
"""

st.markdown(html_temp,unsafe_allow_html=True)

st.header(':wave: Hi Signal Processing Enthusiast !')

st.header('You just hopped on our :satellite_antenna: !')

st.subheader('Overview : This Web Application detects whether the :camera: that you have uploaded is blurry or not based on the algorithm of Fast Fourier Transform (FFT) :bulb:')

image = Image.open(path)

st.image(image, caption='Raw Uploaded Image', use_column_width=True)

result = Image.open('Notblur1result.png')

st.image(result, caption='Grayscale & FFT Results', use_column_width=True)



if st.button("Diagnose"):
    (mean, blurry) = detect_blur_fft(gray, size=60,thresh=10, vis=1)
    st.write('Mean of Magnitude Spectrum : %f' % mean)
    st.write('Blurry Image ? : %s' % blurry)
    

if st.button("About"):
    st.markdown(' Scripted with :yellow_heart: by Sanjeev',unsafe_allow_html=True)
    

st.subheader('Thanks for your time , Appreciate it :palms_up_together: !')


    
        
    




