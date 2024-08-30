import cv2

def replace_background(image, new_background):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    comic_only = cv2.bitwise_and(image, image, mask=mask_inv)
    new_background = cv2.resize(new_background, (image.shape[1], image.shape[0]))
    bg_part = cv2.bitwise_and(new_background, new_background, mask=mask)
    final_image = cv2.add(comic_only, bg_part)
    return final_image
