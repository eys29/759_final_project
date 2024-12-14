
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_number(filename):
    with open(filename, "r") as file:
        text = file.read()

    numbers = re.findall(r'Execution time: (\d+)', text)
    result = list(map(int, numbers))
    result = np.mean(result)
    
    return result

def helper_func():
    # Opencv time
    path = "./opencv-sift"
    small_log = os.path.join(path, "small.txt")
    medium_log = os.path.join(path, "medium.txt")
    large_log = os.path.join(path, "large.txt")
    
    opencv_list = [small_log, medium_log, large_log]

    # Openmp time
    path = "./openmp-sift"
    small_log = os.path.join(path, "small.txt")
    medium_log = os.path.join(path, "medium.txt")
    large_log = os.path.join(path, "large.txt")

    openmp_list = [small_log, medium_log, large_log]
    
    # serial time
    path = "./serial-sift"
    small_log = os.path.join(path, "small.txt")
    medium_log = os.path.join(path, "medium.txt")
    large_log = os.path.join(path, "large.txt")
    
    serial_list = [small_log, medium_log, large_log]
    
    # cuda time
    path = "./cuda-sift"
    small_log = os.path.join(path, "small.txt")
    medium_log = os.path.join(path, "medium.txt")
    large_log = os.path.join(path, "large.txt")
    
    cuda_list = [small_log, medium_log, large_log]
    
    return serial_list, opencv_list, openmp_list, cuda_list
    

if __name__ == '__main__':
    serial_list, opencv_list, openmp_list, cuda_list = helper_func()
    
    