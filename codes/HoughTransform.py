import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

def Canny(img_file, source_path, low_threshold, high_threshold):

    # Obtenha a imagem em tons de cinza
    img_path = source_path + img_file
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detecção de bordas
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_name = f'edges_{img_name}_{low_threshold}_{high_threshold}.jpg'
    final_path = source_path + output_name

    cv2.imwrite(final_path, edges)

    return edges

def Hough(source_path, img_file, edges, rho, theta, threshold, 
          min_line_length, max_line_gap):

    # Obtenha a imagem RGB
    img_path = source_path + img_file
    img = cv2.imread(img_path)
    line_image = np.copy(img)

    # Aplique a transformação de Hough para obter as linhas
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    # Desenhe as linhas na cópia da imagem
    for line in tqdm(lines, desc="Detecting Lines ..."):
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_name = f'hough_{img_name}_{rho}_{theta}_{threshold}_{min_line_length}_{max_line_gap}.jpg'
    final_path = source_path + output_name

    cv2.imwrite(final_path, line_image)
    print(f"Processed image {img_path} ready at:", final_path)

    return lines

