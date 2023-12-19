import cv2
import numpy as np
from PIL import Image
from skimage import filters

def dynamic_expansion(input_image):
    # Define the range of intensities in the starting image
    a0 = np.min(input_image)
    a1 = np.max(input_image)
    # Define the available interval (usually amin=0 and amax=255)
    amin = 0
    amax = 255
    # Calculate the constants 'a' and 'B'
    a = (amin * a1 - amax * a0) / (a1 - a0)
    B = (amax - amin) / (a1 - a0)
    # Apply the dynamic expansion transformation
    modified_image = a + B * (input_image.astype(np.float32))
    # Clip values to ensure they are within [amin, amax] range
    modified_image = np.clip(modified_image, amin, amax)
    # Convert the modified image back to 8-bit unsigned integer format
    modified_image = modified_image.astype(np.uint8)
    
    # Apply CLAHE to the green channel (in this case, the grayscale image)
    enhanced_green = clahe.apply(modified_image)
    # Increase brightness
    brightness_factor = 1.5  # Increase brightnes
    enhanced_green = cv2.convertScaleAbs(enhanced_green, alpha=brightness_factor, beta=0)
    return enhanced_green
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
def maximum_entropy_threshold(image):
    # Compute the histogram of the input image
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Normalize the histogram to obtain probabilities
    histogram /= histogram.sum()
    
    threshold = 0  # Initialize the threshold value
    max_entropy = 0  # Initialize the maximum entropy value

    # Iterate through possible threshold values
    for t in range(1, 256):
        # Calculate probabilities for the object and background classes
        pC1 = np.sum(histogram[:t])
        pC2 = np.sum(histogram[t:])
        
        # Calculate entropies for the object and background classes
        hC1 = -np.sum((histogram[:t] / pC1) * np.log2(histogram[:t] / pC1 + 1e-10))  # Adding a small constant to avoid log(0)
        hC2 = -np.sum((histogram[t:] / pC2) * np.log2(histogram[t:] / pC2 + 1e-10))  # Adding a small constant to avoid log(0)
        
        # Calculate the total entropy for the current threshold
        total_entropy = hC1 + hC2
        
        # Update the threshold if we found a higher entropy value
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            threshold = t

    return threshold
def extract_bv(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    #Contrast Limited Adaptive Histogram Equalization is applied
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))

    #LAB modal converted back to RGB
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('Canny Edges', edges)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Applying alternate sequential filtering (3 times closing opening)
    blue,green,red = cv2.split(final)
    r1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23,23)), iterations = 1)
    f4 = cv2.subtract(R3, green)
    f5 = clahe.apply(f4)

    #tophat morphological transformation
    image1 = f5
    e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closeImg = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, e_kernel)
    revImg = closeImg
    topHat = image1 - revImg

    #otsu with probability and minimization function
    imge = topHat
    blur = cv2.GaussianBlur(imge, (5,5), 0)
    hist = cv2.calcHist([blur], [0], None, [256], [0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i]) #probabilities
        q1, q2 = Q[i],Q[255]-Q[i] #cum sum of classes
        b1, b2 = np.hsplit(bins,[i]) #weights

    #finding means and variances
    if q1 == 0:
        q1 = 0.0000001
    if q2 == 0:
        q2 = 0.0000001
    m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1, v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    #calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

    #find otsu&'s threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY) 
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255   
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)            
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)            
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)   

    # Removing blobs of unwanted bigger chunks
    im_eroded = cv2.bitwise_not(newfin) 
    xmask = np.ones(im.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(im_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)                  
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"    
        else:
            shape = "veins"
        if shape == "circle":
            cv2.drawContours(xmask, [cnt], -1, 0, -1)   

    finimage = cv2.bitwise_and(im_eroded, im_eroded, mask=xmask)  
    blood_vessels = cv2.bitwise_not(finimage)
    blood_vessels = cv2.subtract(255, blood_vessels)
    
    return blood_vessels
def ExudInp(I, ExudMask, radius):
    PxToFill = ExudMask - cv2.erode(ExudMask, np.ones((radius,radius), np.uint8))
    coordinates = np.column_stack(np.where(PxToFill != 0))
    
    for p in coordinates:
        x, y = p
        x1 = max(x - radius, 0)
        x2 = min(x + radius, I.shape[0] - 1)
        y1 = max(y - radius, 0)
        y2 = min(y + radius, I.shape[1] - 1)

        Np = I[x1:x2+1, y1:y2+1]
        mean_val = np.mean(Np[Np != 0])
        I[x, y] = mean_val

def red_lesion_segmentation(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV for mạch máu
    lower_blood_vessel = np.array([0, 50, 50])  
    upper_blood_vessel = np.array([10, 255, 255])  

    # Create a mask to extract blood vessels
    blood_vessel_mask = cv2.inRange(hsv_image, lower_blood_vessel, upper_blood_vessel)

    # Define the range of red color in HSV for tổn thương
    middle_value = 300

    # Calculate the color range based on the desired width (6) around the middle value
    color_range = 6 // 2
    lower_lesion = np.array([middle_value - color_range, 50, 50])
    upper_lesion = np.array([middle_value + color_range, 255, 255])

    # Create a mask to extract lesions
    lesion_mask = cv2.inRange(hsv_image, lower_lesion, upper_lesion)
    # Combine the masks
    combined_mask = cv2.bitwise_or(blood_vessel_mask, lesion_mask)

    # Apply morphological operations to enhance the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Bitwise AND the original image with the mask to get the segmented red lesions
    segmented_image = cv2.bitwise_and(img, img, mask=combined_mask)

    # Create a binary mask for the lesion area
    lesion_area_mask = cv2.inRange(segmented_image, (0, 0, 1), (255, 255, 255))

    # Set the lesion area to white in the original image
    img[np.where(lesion_area_mask)] = [0, 0, 0]

    return lesion_area_mask
#---------------------MAIN-----------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# Đọc ảnh và tiền xử lý
OrgImg = cv2.imread(r'D:\document\95IDEAL\code\DR_detection\OpenCV\images\vang.png')
OrgImg = cv2.resize(OrgImg,(550,380))
green_channel = OrgImg[:,:,1]

blood_vessels = extract_bv(OrgImg)
cv2.imshow('blood_vessels',blood_vessels)
gray = cv2.cvtColor(OrgImg,cv2.COLOR_BGR2GRAY)
median_filter_size = 69
Imed = cv2.medianBlur(gray, median_filter_size)
D = cv2.subtract(gray, Imed)
IC = cv2.normalize(D, None, 0, 255, cv2.NORM_MINMAX)
expanded_image = dynamic_expansion(IC)
output_image = cv2.medianBlur(expanded_image, 3)
dst2 = cv2.inpaint(output_image, blood_vessels, 5, cv2.INPAINT_NS)

# Phát hiện vùng exudate
threshold_value = maximum_entropy_threshold(dst2)

_, ExudMask = cv2.threshold(dst2, threshold_value, 255, cv2.THRESH_BINARY)

# Tạo bản sao của ExudMask để theo dõi
ExudMaskCopy = np.copy(ExudMask)


# Tìm contours trong ảnh nhị phân
contours, _ = cv2.findContours(ExudMaskCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tìm contour có diện tích lớn nhất
max_area = 0
max_contour_index = -1

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour_index = i

# Tạo ảnh trắng mới
mask = np.zeros_like(ExudMaskCopy)

# Tạo ảnh đen
result_image = np.zeros_like(OrgImg)

# Vẽ vùng có diện tích lớn nhất lên ảnh đen
cv2.drawContours(result_image, contours, max_contour_index, (255, 255, 255), thickness=cv2.FILLED)

# Vẽ lại tất cả contour trừ contour có diện tích lớn nhất
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
cv2.drawContours(mask, [contours[max_contour_index]], -1, (0), thickness=cv2.FILLED)
# Loại bỏ contour có diện tích lớn nhất khỏi ảnh nhị phân
exduate = cv2.bitwise_and(ExudMaskCopy, mask)
mask_resized = cv2.resize(mask, (OrgImg.shape[1], OrgImg.shape[0]))
mask_inverted = cv2.bitwise_not(mask_resized)

# Create a mask for blood vessels
blood_vessels_mask = blood_vessels.copy()

# Threshold or use segmentation to create the mask for blood vessels
# Example: You may use a simple thresholding
_, blood_vessels_mask = cv2.threshold(blood_vessels_mask, 128, 255, cv2.THRESH_BINARY)

# Invert the blood vessels mask
blood_vessels_mask_inv = cv2.bitwise_not(blood_vessels_mask)

OrgImg_removed = cv2.bitwise_and(OrgImg, OrgImg, mask=mask_inverted)
OrgImg_removed = cv2.bitwise_and(OrgImg_removed, OrgImg_removed, mask=blood_vessels_mask_inv)
cv2.drawContours(OrgImg_removed, contours, max_contour_index, (0, 0, 0), thickness=cv2.FILLED)

green_channel_removed = OrgImg_removed[:,:,1]

# Enhance Image using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Apply CLAHE to the green channel
enhanced_green_new = clahe.apply(green_channel_removed)
# Create a blank image with the same size as the original image
enhanced_image_new = cv2.merge((OrgImg_removed[:,:,0], enhanced_green_new, OrgImg_removed[:,:,2]))
# Display the enhanced image
red_lesion = red_lesion_segmentation(enhanced_image_new)


# Hiển thị ảnh gốc và ảnh kết quả
cv2.imshow('Original Image', OrgImg)
cv2.imshow('Exudate Image', exduate)
cv2.imshow('OD Image', result_image)
cv2.imshow('red_lesion', red_lesion)
OrgImg_removed[np.where(red_lesion)] = [0, 0, 0]
cv2.imshow('Segmentation',OrgImg_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()