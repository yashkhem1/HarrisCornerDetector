import numpy as np
import pandas as pd
import os,sys
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import argparse

class HarrisCornerDetector(object):
    def __init__(self,k,threshold,sigma1,sigma2):
        """
        Args:
            k (float): Parameter used for calculating cornerness measure
            threshold (float): Cornerness threshold
            sigma1 (float): Sigma parameter for Gaussian Filter used for smoothening the image
            sigma2 (float): Sigma parameter for Gaussian Filter used for Window function
        """
        self.k = k
        self.threshold = threshold
        self.sigma1 = sigma1
        self.sigma2 = sigma2


    def initialize_image(self,imgpath):
        """Preprocessing of the image

        Args:
            imgpath (string): Path to input image
        """
        self.imgname = os.path.basename(imgpath)
        self.img = Image.open(imgpath)
        self.processed_img = np.array(self.img.convert("L")) #Converting to grayscale
        self.processed_img = self.processed_img/np.max(self.processed_img) #Normalize the range to (0,1)
        self.processed_img = gaussian_filter(self.processed_img,self.sigma1) #Smooth image to filter out any noise

    def get_cornerness_measure(self):
        """Create the structure tensor matrix
        """
        grad_y, grad_x = np.gradient(self.processed_img) #Getting the gradients for the image
        Ixx = gaussian_filter(grad_x**2,self.sigma2) #Gaussian Filtering here acts as the window function
        Iyy = gaussian_filter(grad_y**2,self.sigma2)
        Ixy = gaussian_filter(grad_x*grad_y,self.sigma2)

        determinant_A = Ixx*Iyy - Ixy**2
        trace_A = Ixx+Iyy
        self.cornerness = determinant_A - self.k*trace_A**2

    def nms_and_threshold(self):
        """Non-maximal suppression and thresholding
        """
        #NMS
        nrows,ncols = self.processed_img.shape
        for i in range(2,nrows-1):
            for j in range(2,ncols-1):
                max_cornerness = np.max(self.cornerness[i-1:i+1,j-1:j+1])
                if self.cornerness[i,j] != max_cornerness:
                    self.cornerness[i,j] = 0

        #Thresholding
        self.cornerness[self.cornerness<self.threshold] = 0

    def show_corners(self,output_dir=""):
        """Show the corners of the image

        Args:
            output_dir (str): Path to output directory
        """
        show_img = np.array(self.img)[:,:,:3]
        #If greyscale, convert image to rgb
        if len(show_img.shape)==2:
            show_img = np.expand_dims(show_img,2)
        if show_img.shape[2] == 1:
            show_img = np.repeat(show_img,3,2)

        #Mark the corner points with red color
        x_coords, y_coords = np.where(self.cornerness>0)
        plt.imshow(show_img)
        plt.plot(y_coords,x_coords,'*', color='red')
        if output_dir:
            plt.savefig(os.path.join(output_dir,self.imgname))
        plt.title("Harris Corner Detector Output")
        plt.show()
        plt.figure()
        plt.imshow(self.cornerness)
        plt.title("Cornerness measure")
        plt.show()

    def detect_corners(self,imgpath,output_dir):
        """Detect the corners of the input image. Just an order-wise calling of the above functions

        Args:
            imgpath (str): Path to input image
        """
        self.initialize_image(imgpath)
        self.get_cornerness_measure()
        self.nms_and_threshold()
        self.show_corners(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath",type=str,nargs="+",help="Path to input image(s)")
    parser.add_argument("--k",type=float,help="Parameter used for calculating cornerness measure")
    parser.add_argument("--threshold",type=float,help="Cornerness Threshold")
    parser.add_argument("--sigma1",type=float,help="Sigma parameter for Gaussian Filter used for smoothening the image")
    parser.add_argument("--sigma2",type=float,help="Sigma parameter for Gaussian Filter used for Window function")
    parser.add_argument("--output_dir",type=str,default="",help="Path to output dir")
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
    hcd = HarrisCornerDetector(args.k,args.threshold,args.sigma1,args.sigma2)
    for imgpth in args.imgpath:
        hcd.detect_corners(imgpth,args.output_dir)

