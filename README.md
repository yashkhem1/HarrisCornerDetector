# Harris Corner Detection in Python
## Yash Khemchandani(170050025), Abhijeet Singh Yadav(170050021), Manas Vashistha(17D070064)

In this repository we have implemented Harris Corner Detection from scratch using Pyton. The code is available in <i>harris_corner.py</i> file. To get the Harris Corner output of an image, run:
```
python3 harris_corner.py --k <k> --sigma1 <sigma1> --sigma2 <sigma2> --threshold <threshold> --imgpath <path_to_input_image> --output_dir <path_to_output_dir>
```
where
- \<k\> : Hyperparameter used in calculating the cornerness measure/ Harris response at each pixel
- \<sigma1\> : The standard deviation of Gaussian Filter used for smoothening the image
- \<sigma2\> : The standard deviation of Gaussian Filter which is used as the window function for computing the Structure Matrix
- \<threshold\> : Used for final thresholding of the cornerness measure

Sample images are present in the <b>input</b> folder and their corresponding Harris Corner outputs are present in the <b>output</b> folder. To generate these exact outputs, run `bash scripts.sh`

