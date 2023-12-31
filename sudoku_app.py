#! /usr/bin/env python3
#  ~*~ coding:utf-8 ~*~

# Name: Sudoku solver application
# Description: this application is design to be deployed on Streamlit Cloud so that it can be accessible by anyone.

# Author: Alexandre Flageul
# Sources: 
#   - preprocessing of the input image: https://www.kaggle.com/code/karnikakapoor/sudoku-solutions-from-image-computer-vision
#   - sudoku solver python code: https://github.com/aurbano/sudoku_py
#   - perspective of evolution: https://icosys.ch/sudoku-dataset

## modules

import streamlit as st
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
#import split_image
import numpy as np
import os
import cv2
from datetime import datetime
from models.sudoku_solver import sudoku_solver
import matplotlib.pyplot as plt



## functions

def standardize_picture(image):

    def preprocess(image):
        """to greyscale, blur and change the receptive threshold of image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (3,3),6) 
        #blur = cv2.bilateralFilter(gray,9,75,75)
        threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        return threshold_img


    def main_outline(contour):
        biggest = np.array([])
        max_area = 0
        for i in contour:
            area = cv2.contourArea(i)
            if area >50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i , 0.02* peri, True)
                if area > max_area and len(approx) ==4:
                    biggest = approx
                    max_area = area
        return biggest, max_area


    def reframe(points):
        points = points.reshape((4, 2))
        points_new = np.zeros((4,1,2), dtype=np.int32)
        add = points.sum(1)
        points_new[0], points_new[3] = points[np.argmin(add)], points[np.argmax(add)]
        diff = np.diff(points, axis =1)
        points_new[1], points_new[2] = points[np.argmin(diff)], points[np.argmax(diff)]
        return points_new


    def splitcells(img):
        rows = np.vsplit(img,9)
        return [ box for r in rows for box in np.hsplit(r, 9) ]


    def CropCell(cells):
        Cells_croped = []
        for image in cells:
            img = np.array(image)
            img = Image.fromarray(img).crop((5, 5, 44, 44)).resize((28,28))
            img = img.convert("L")
            img = np.array(img)
            Cells_croped.append(img)
        
        return Cells_croped

    # loading image and resize it
    sudoku_a = cv2.imread(image)
    sudoku_a = cv2.resize(sudoku_a, (450,450))

    # preprocess image
    threshold = preprocess(sudoku_a)

    # Finding the outline of the sudoku puzzle in the image
    contour_1, contour_2 = sudoku_a.copy(), sudoku_a.copy()
    contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contour, -1, (0,255,0), 3)


    black_img = np.zeros((450,450,3), np.uint8)
    biggest, maxArea = main_outline(contour)
    if biggest.size != 0:
        biggest = reframe(biggest)
        cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)  
        imagewrap = cv2.warpPerspective(sudoku_a,matrix,(450,450))
        #imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
        imagewrap = preprocess(imagewrap)

        # split standardized sudoku image
        sudoku_cell = splitcells(imagewrap)

        # crop sudoku cells
        return CropCell(sudoku_cell) # list of numpy array of size (28,28)
    else: raise Exception("Problem with input image")


def turn_list_of_array_to_matrice_of_integer(list_of_array):
    # load digits identification model
    path_to_model_digits_identification = "models/new_digits_identification_model.h5"
    model_identify_digits = tf.keras.models.load_model(path_to_model_digits_identification)

    cases = []

    for i in list_of_array:
        #plt.figure()
        #plt.imshow(i)
        #plt.show()

        if np.mean(i) < 26 : value = 0
        else:
            pred = model_identify_digits.predict(i.reshape(1,28,28,1))
            value = np.argmax(pred)
            probability = np.max(pred[0])
            #print(value, probability, np.mean(i))
        cases.append(value)
        
        #st.write("--------------------")
        #st.write(np.mean(i))
        #st.image(i)
        #st.write(value)
    
    # now, we have a sudoku that is a list of digits between 0 and 9, we convert into numpy array with a specific shape
    return np.array(cases).reshape(9, 9)



def generate_sudoku_solution_image(sudoku_solution_array):

    oneliner_sudoku_solution = "".join([str(value) for row in sudoku_solution_array for value in row ])
    
    pixel = 860
    image = Image.new("RGB",size=(pixel+10, pixel+10), color="black")
    bloc_size = int((pixel/9)-5)
    bloc = Image.new("RGB", size=(bloc_size, bloc_size), color="white")

    rows, columns = range(9), range(9)
    font = ImageFont.truetype("font/times-new-roman.ttf", 75)

    for r in rows:
        for c in columns:
            # ces deux boucles permettent d'ajouter les cases blanches et le texte de gauche à droite et de haut en bas
            image.paste(bloc, (10+c*95, 10+r*95))
            value = oneliner_sudoku_solution[9*r + c]
            ImageDraw.Draw(image).text((35+c*95, 10+r*95), value, (0, 0, 0), font=font, )

    return image

def solving_sudoku(sudoku_array):
    try:
        solution = sudoku_solver(sudoku_array)
        return solution
    except Exception:
        st.write("--------------------")
        st.write("It seams that no solution were found, probably it's because ilage analysis fail to correctly identify digits, Please try again.")
        st.write("--------------------")
        return 


## General structure

# starting the app
path_to_icon = ""
st.set_page_config(page_title="sudoku_solver", layout="wide", page_icon=path_to_icon)

st.title("Sudoku solver!!")
st.header("Take a picture, and that's it!")

st.divider()
label = "Take a picture of your sudoku"
picture = st.camera_input(label, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
st.divider()

if picture != None:
    # to generate unique folder name
    out = "".join(str(datetime.now()).replace("-", " ").replace(":"," ").replace(".", " ").split(" "))
    output_dir = f"working_directory/{out}/"

    # save image into folder working_directory
    img = Image.open(picture)
    image_name = f"{output_dir[:-1]}.png"
    img.save(image_name, image_name.split(".")[-1].upper())

    # get the picture, extract sudoku and turn it into oneliner
    processed_image = standardize_picture(image_name) # return a list of image in numpy array, each with a size of (28, 28)
                                # to not modify the next step, it should return an Image of size 860*860

    # turn sudoku standardized image into oneliner sudoku
    array_sudoku = turn_list_of_array_to_matrice_of_integer(processed_image)

    # solve sudoku
    ## pure python solving sudoku
    sudoku_solution = solving_sudoku(array_sudoku) # return a numpy array of the solution dimension (9x9)
    
    # convert oneliner solution into image
    image_solution = generate_sudoku_solution_image(sudoku_solution)

# display image
if picture != None:
    st.divider()
    label = "Here is the solution"
    st.header(label)
    st.image(image_solution)
    st.divider()


