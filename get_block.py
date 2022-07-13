import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from copy import deepcopy
from keras.models import load_model
import os

# Contains the file path for a sudoku board
file = r'path'



# Iterates through a Sudoku Board
# Gets Each block and saves into file
# Passes block_path to NN to get value
# Removes saved image

# NOTE:
# There is probably a more efficient way in doing this. For instance, most likely did not have to 
# save each block into file, but I just preferred to. 
def get_block(file):
    # Folder to save blocks
    path = r'------------'
    # Path where model is saved
    file_path = r'--------------'
    
    # Load model from file
    model = load_model(file_path)
    
    
    
    # Convert board into 252x252 pixels
    # Convert board to grayscale
    img = cv2.imread(file)
    img = cv2.resize(img, (252, 252))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create Threshold Image
    # Image is now binary. Since some horizontal and vertical lines disappear depending on how the image is taken
    # We erode.
    
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh_two = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    cnts = cv2.findContours(thresh_two, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    
 
    # Get horizontal and vertical lines
    #vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    #thresh_two = cv2.morphologyEx(thresh_two, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    
    # Draw horizontal and vertical lines on board since some get eroded away when converting to binary image
    # This helps make sure each image can be easily read by NN without changes
    horizontal = np.copy(thresh_two)
    vertical = np.copy(thresh_two)

    hor_kernel = np.ones((1, 20), np.uint8)
    horizontal = cv2.erode(horizontal, hor_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, hor_kernel, iterations=9)
    
    vert_kernel = np.ones((20, 1), np.uint8)
    vertical = cv2.erode(vertical, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=9)
    
    res = vertical + horizontal


    # Combines drawn vertical and horizontal lines on board
    fin = cv2.addWeighted(thresh, 1, res, 1, 0)

    
    cnts = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    min_area = 500
    max_area = 1000
    image_number = 0
    digits = []
    values = []
    
    
    # Using Contours, if the area is less < max and > min, there must be a block there
    # Take the coordinates the drawn reactangle, and iterate through
    # Then we resize into 28x28 (since this is the image our NN was trained on)
    # To make sure ONLY the number exists (remove noise), we mask it onto a blank 28x28 block
    # Then save into file and pass into NN to get a 2D array of our board
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            
            # Iterate through each block            
            ROI = fin[y:y+h, x:x+w]
            ROI = cv2.resize(ROI, (28, 28))

            # Create blank 28x28 image to mask the block onto
            img_black = np.zeros((28, 28), np.uint8)
            cv2.rectangle(img_black, (4, 4), (22, 22), 255, -1)
            masked = cv2.bitwise_and(ROI, ROI, mask=img_black)

            # Store the image into file
            cv2.imwrite(os.path.join(path, f'ROI_{image_number}.png'), masked)
            
            # Evaluate image
            value = os.listdir(path)
            digits = evaluate(os.path.join(path, value[0]))
            values.append(digits)

            # Remove image from file
            os.remove(os.path.join(path, value[0]))
            
            image_number += 1
    
    return construct_board(values)



# Takes image data from file, passes into NN, and creates a 2D array depending on the predicted value
def construct_board(values):
    print(len(values))
    board = np.zeros((9, 9), dtype=str)
    i = 0
    for r, row in enumerate(board):
        for c, col in enumerate(board):
            board[r,c] = str(values[i])
            i += 1            
    
    return np.flip(board)

"""
array([['-', '-', '-', '8', '9', '-', '4', '-', '-'],
       ['-', '2', '8', '-', '-', '5', '-', '-', '3'],
       ['-', '-', '1', '-', '3', '4', '-', '-', '8'],
       ['9', '8', '-', '-', '-', '1', '-', '7', '-'],
       ['2', '-', '7', '-', '6', '9', '-', '3', '1'],
       ['-', '-', '3', '2', '-', '7', '5', '4', '-'],
       ['8', '3', '-', '9', '-', '-', '-', '-', '7'],
       ['-', '-', '5', '-', '2', '-', '-', '-', '-'],
       ['1', '-', '6', '5', '7', '3', '-', '-', '4']], dtype='<U1')
"""

def clear_file(file):
    for f in os.listdir(file):
        os.remove(os.path.join(file, f))


#clear_file(r'C:\Users\Memo\Pictures\sudoku AI\Board Numbers')
get_block(file)
#construct_board(values)