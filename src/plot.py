
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

if __name__ == '__main__':
    small_log = os.path.join("./opencv-sift", "small.txt")
    medium_log = os.path.join("./opencv-sift", "medium.txt")
    large_log = os.path.join("./opencv-sift", "large.txt")
    
    small_time = extract_number(small_log)
    medium_time = extract_number(medium_log)
    large_time = extract_number(large_log)
    
    print(small_time, medium_time, large_time)
    