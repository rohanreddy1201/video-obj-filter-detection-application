
import cv2
import numpy as np
import sys

# Define matrices for color space transformations
# Matrix for LMS to RGB conversion
lms_to_rgb_matrix = np.array([
    [4.4679, -3.5873, 0.1193],
    [-1.2186, 2.3809, -0.1624],
    [0.0497, -0.2439, 1.2045]
])

# Matrix for RGB to LMS conversion (inverse of LMS to RGB)
rgb_to_lms_matrix = np.linalg.inv(lms_to_rgb_matrix)

# Matrix for LMS to Lab conversion
lms_to_lab_matrix = np.array([
    [1/np.sqrt(3), 0, 0],
    [0, 1/np.sqrt(6), 0],
    [0, 0, 1/np.sqrt(2)]
]) @ np.array([
    [1, 1, 1],
    [1, 1, -2],
    [1, -1, 0]
])

# Matrix for Lab to LMS conversion (inverse of LMS to Lab)
lab_to_lms_matrix = np.linalg.inv(lms_to_lab_matrix)

# Matrix for LMS to CIECAM97s conversion
lms_to_ciecam97s_matrix = np.array([
    [2.00, 1.00, 0.05],
    [1.00, -1.09, 0.09],
    [0.11, 0.11, -0.22]
])

# Matrix for CIECAM97s to LMS conversion (inverse of LMS to CIECAM97s)
ciecam97s_to_lms_matrix = np.linalg.inv(lms_to_ciecam97s_matrix)

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    img_RGB = img_BGR[:, :, ::-1]  # RGB to BGR is just reversing the channels
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB, dtype=np.float32)
    img_BGR = img_RGB[:, :, ::-1]  # RGB to BGR is just reversing the channels
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    # Convert to LMS space
    flat_RGB = img_RGB.reshape(-1, 3)
    flat_LMS = np.dot(flat_RGB, rgb_to_lms_matrix.T)
    # Logarithmic nonlinearity
    with np.errstate(divide='ignore'):
        flat_LMS = np.where(flat_LMS > 0, np.log10(flat_LMS), 0)
    # Convert to Lab space
    img_Lab = np.dot(flat_LMS, lms_to_lab_matrix.T).reshape(img_RGB.shape)
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    # Convert to LMS space
    flat_Lab = img_Lab.reshape(-1, 3)
    flat_LMS = np.dot(flat_Lab, lab_to_lms_matrix.T)
    # Exponential nonlinearity
    flat_LMS = np.power(10, flat_LMS)
    # Convert to RGB space
    img_RGB = np.dot(flat_LMS, lms_to_rgb_matrix.T).reshape(img_Lab.shape)
    # Clip to valid range and cast to uint8
    img_RGB = np.clip(img_RGB, 0, 255).astype(np.uint8)
    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    # Convert to LMS space
    flat_RGB = img_RGB.reshape(-1, 3)
    flat_LMS = np.dot(flat_RGB, rgb_to_lms_matrix.T)
    # Logarithmic nonlinearity
    with np.errstate(divide='ignore'):
        flat_LMS = np.where(flat_LMS > 0, np.log10(flat_LMS), 0)
    # Convert to CIECAM97s space
    img_CIECAM97s = np.dot(flat_LMS, lms_to_ciecam97s_matrix.T).reshape(img_RGB.shape)
    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    # Convert to LMS space
    flat_CIECAM97s = img_CIECAM97s.reshape(-1, 3)
    flat_LMS = np.dot(flat_CIECAM97s, ciecam97s_to_lms_matrix.T)
    # Exponential nonlinearity
    flat_LMS = np.power(10, flat_LMS)
    # Convert to RGB space
    img_RGB = np.dot(flat_LMS, lms_to_rgb_matrix.T).reshape(img_CIECAM97s.shape)
    # Clip to valid range and cast to uint8
    img_RGB = np.clip(img_RGB, 0, 255).astype(np.uint8)
    return img_RGB

def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    source_Lab = convert_color_space_RGB_to_Lab(img_RGB_source)
    target_Lab = convert_color_space_RGB_to_Lab(img_RGB_target)

    # Compute source and target means and standard deviations for each channel
    mean_src = np.mean(source_Lab, axis=(0, 1))
    std_src = np.std(source_Lab, axis=(0, 1))
    mean_tgt = np.mean(target_Lab, axis=(0, 1))
    std_tgt = np.std(target_Lab, axis=(0, 1))

    # Subtract the means from the source image's color channels
    normalized_src = source_Lab - mean_src

    # Scale by the standard deviations ratio (target over source) for each channel
    # Avoid division by zero in case std_src is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_factors = std_tgt / std_src
        scale_factors[std_src == 0] = 0
        scaled_src = normalized_src * scale_factors

    # Add the target's mean to the scaled source image's color channels
    transferred_Lab = scaled_src + mean_tgt

    # Convert the transferred Lab image back to the RGB color space
    transferred_RGB = convert_color_space_Lab_to_RGB(transferred_Lab)
    
    return transferred_RGB

def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # Convert the source and target images to float32 for precision
    img_RGB_source = img_RGB_source.astype(np.float32)
    img_RGB_target = img_RGB_target.astype(np.float32)

    # Calculate means and standard deviations of the source and target images
    mean_src, std_src = np.mean(img_RGB_source, axis=(0, 1)), np.std(img_RGB_source, axis=(0, 1))
    mean_tgt, std_tgt = np.mean(img_RGB_target, axis=(0, 1)), np.std(img_RGB_target, axis=(0, 1))

    # Subtract the mean from the source image
    normalized_src = img_RGB_source - mean_src
    
    # Prevent division by zero by setting std_src to 1 where it is 0
    std_src_corrected = np.where(std_src == 0, 1, std_src)

    # Scale the normalized source image by the standard deviation ratio
    scaled_src = normalized_src * (std_tgt / std_src_corrected)
    
    # Add the target mean
    transferred_src = scaled_src + mean_tgt

    # Clip the values to maintain the valid RGB range and convert back to uint8
    transferred_src = np.clip(transferred_src, 0, 255).astype(np.uint8)

    return transferred_src

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    # Convert source and target RGB images to CIECAM97s color space
    source_CIECAM97s = convert_color_space_RGB_to_CIECAM97s(img_RGB_source)
    target_CIECAM97s = convert_color_space_RGB_to_CIECAM97s(img_RGB_target)

    # Calculate means and standard deviations of the source and target images in CIECAM97s color space
    mean_src, std_src = np.mean(source_CIECAM97s, axis=(0, 1)), np.std(source_CIECAM97s, axis=(0, 1))
    mean_tgt, std_tgt = np.mean(target_CIECAM97s, axis=(0, 1)), np.std(target_CIECAM97s, axis=(0, 1))

    # Subtract the mean from the source CIECAM97s image
    normalized_src = source_CIECAM97s - mean_src

    # Prevent division by zero by setting std_src to 1 where it is 0
    std_src_corrected = np.where(std_src == 0, 1, std_src)

    # Scale the normalized source image by the standard deviation ratio
    scaled_src = normalized_src * (std_tgt / std_src_corrected)
    
    # Add the target mean
    transferred_src = scaled_src + mean_tgt

    # Convert the transferred CIECAM97s image back to RGB color space
    transferred_src_RGB = convert_color_space_CIECAM97s_to_RGB(transferred_src)

    # Clip the values to maintain the valid RGB range and convert back to uint8
    transferred_src_RGB = np.clip(transferred_src_RGB, 0, 255).astype(np.uint8)

    return transferred_src_RGB

def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

def rmse(apath,bpath):
    """
    This is the help function to get RMSE score.
    apath: path to your result
    bpath: path to our reference image
    when saving your result to disk, please clip it to 0,255:
    .clip(0.0, 255.0).astype(np.uint8))
    """
    a = cv2.imread(apath).astype(np.float32)
    b = cv2.imread(bpath).astype(np.float32)
    return np.sqrt(np.mean((a-b)**2))


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Spring 2024, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]
    
    # ===== read input images
    img_RGB_source = cv2.imread(path_file_image_source).astype(np.float32)
    img_RGB_target = cv2.imread(path_file_image_target).astype(np.float32)
    
    # Convert and save the image in Lab color space
    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8))

    # Convert and save the image in RGB color space
    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB.clip(0.0, 255.0).astype(np.uint8))

    # Convert and save the image in CIECAM97s color space
    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s.clip(0.0, 255.0).astype(np.uint8))

    # RMSE computation (Please change 'result1.png' as per reference image)
    print("RMSE in Lab = ", rmse(path_file_image_result_in_Lab, 'result1.png'))
    print("RMSE in RGB = ", rmse(path_file_image_result_in_RGB, 'result1.png'))
    print("RMSE in CIECAM97s = ", rmse(path_file_image_result_in_CIECAM97s, 'result1.png'))