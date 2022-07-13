# sudoku-ai


## Description
* Purpose of this project is to see if I can apply my OpenCV knowledge into the real-world. Program uses OpenCV to take a sudoku board and break it into individual pieces. Then using a Convolutional Neural Network trained on the MNIST dataset, convert each block into a 2D array. Pass the 2D array into an algorithm that uses brute force to solve. 

* A challenge I faced was breaking the entire board into their respective blocks to pass into the Neural Network. However, after enough trial-and-errors, I realized I could just break the contours up into 9x9 blocks. 

