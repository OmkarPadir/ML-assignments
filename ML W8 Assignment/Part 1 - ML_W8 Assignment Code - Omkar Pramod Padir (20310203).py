# Name: Omkar Pramod Padir
# Student Id: 20310203
# Course: Machine Learning CS7CS4
# Week 8 Assignment Part1


import numpy as np

# Part i-a starts here

def Convolute(input_arr, kernel):

    result_arr = []

    # Define step size
    STRIDE = 1

    # Get the dimensions of input and kernel
    i_m = len(input_arr)
    i_n = len(input_arr[0])

    k_m = len(kernel)
    k_n = len(kernel[0])

    # loop through individual values and perform convolution to get output array
    temp_step_i=0
    for x in range (i_m-k_n+1):

        temp_arr=[]
        temp_step_j=0
        for y in range (i_n - k_n+1):

            temp_res = 0
            for i in range(k_m):
                for j in range(k_n):

                    inp_val = input_arr[i+temp_step_i][j+temp_step_j]
                    ker_val = kernel[i][j]

                    res_val = ker_val*inp_val

                    temp_res=temp_res+res_val

            temp_arr.append(temp_res)
            temp_step_j=temp_step_j+STRIDE

        result_arr.append(temp_arr)
        temp_step_i = temp_step_i+STRIDE


    return result_arr


# Part i - b starts here

from PIL import Image

im = Image.open('Images\Shop.png') # Shop.png
rgb = np.array(im.convert('RGB'))
r = rgb [:,:,0] # array of R pixels

# print(r)
# print(rgb.shape)
# exit(1)
Image.fromarray(np.uint8(r)).show()
Image.fromarray(np.uint8(r)).save("Images\Shop_red.png")

# Define Kernels
K1 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
K2 = [[0,-1,0],[-1,8,-1],[0,-1,0]]

# Output of rgb array after convolution ; parameters are inputArray, Kernel and Stride
res1 = Convolute(r,K1)
Image.fromarray(np.uint8(res1)).show()
Image.fromarray(np.uint8(res1)).save("Images\Shop_K1.png")

res2 = Convolute(r,K2)
Image.fromarray(np.uint8(res2)).show()
Image.fromarray(np.uint8(res2)).save("Images\Shop_K2.png")