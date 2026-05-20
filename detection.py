import cv2
import numpy as np
from pathlib import Path

# folder where the test images are stored
IMAGE_DIR = Path("test images")
OUTPUT_DIR = Path("detected_images")

# these values worked fine for the test images, so if the actual images are similar, they should work fine. 
# #If not, they can be adjusted.
# UPDATE: i increased the min area from 300 to 1000 to filter out the small red regions that were being detected as circles
MIN_AREA = 1000

#circularity is a measure of how circular an object is, with 1 being a perfect circle 
#and values closer to 0 being more irregular.
#so if the detection fails because it is filtering out the red circles, 
# we can try lowering since it would increase tolerance but may introduce false positives from non-circular shapes.
MIN_CIRCULARITY = 0.75

#also i wanted to increase this threshold from 0.7 to 0.8 to avoid detecting messy shapes that are not fully circles
MIN_FILL_RATIO = 0.7

DEBUG = False


#criteria for determining if an image contains a red circle:
#1. HSV red range
#2. Minimum contour area
#3. Circularity threshold
#4. Fill ratio threshold (to filter out hollow rings)
def contains_red_circle(img, image_name="unknown"):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #ERROR HANDLING#
    if DEBUG:
        print(f"[{image_name}] DEBUG: Image converted to HSV")
    ################       


#color thresholds for red in HSV color space - defines what count as "red". 
#so if the detections fails, it could be because
    # --> the red is darker -> lower the V minimum value in the lower_red arrays
    # --> red is washed out -> lower the S minimum value in the lower_red arrays
    # --> red slightly off -> widen the H range in the lower_red and upper_red arrays to capture a wider range of red hues.
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    #ERROR HANDLING#
    if DEBUG:
        red_pixels = cv2.countNonZero(mask)
        print(f"[{image_name}] DEBUG: Number of red pixels in mask: {red_pixels}")
        if red_pixels == 0:
            print(f"[{image_name}] DEBUG: No red pixels detected in the mask. Consider adjusting the HSV thresholds.")
    ################  


    #create a small 5x5 kernel to help clean up the mask
    kernel = np.ones((5, 5), np.uint8)

    #use morphology close to fill small gaps and connect nearby red areas
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    #find the outer contours from the cleaned mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    closest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 20:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        if circle_area == 0:
            continue

        fill_ratio = area / circle_area

        area_ok = area >= MIN_AREA
        circularity_ok = circularity >= MIN_CIRCULARITY
        fill_ok = fill_ratio >= MIN_FILL_RATIO

        #keep track of the biggest contour just for debugging
        if area > largest_area:
            largest_area = area
            closest_contour = (area, circularity, fill_ratio, area_ok, circularity_ok, fill_ok)

        if area_ok and circularity_ok and fill_ok:
            if DEBUG:
                print(f"[{image_name}] DEBUG: looks like a valid red circle")
            return True

    #if nothing passed, show info for the biggest contour
    if DEBUG and closest_contour is not None:
        area, circularity, fill_ratio, area_ok, circularity_ok, fill_ok = closest_contour
        print(f"[{image_name}] DEBUG (Biggest shape it found):")
        print(f"    Area: {area} (>= {MIN_AREA}? {area_ok})")
        print(f"    Circularity: {circularity} (>= {MIN_CIRCULARITY}? {circularity_ok})")
        print(f"    Fill ratio: {fill_ratio} (>= {MIN_FILL_RATIO}? {fill_ok})")

    if DEBUG:
        print(f"[{image_name}] DEBUG: nothing passed all the checks")

    return False


def main():
    image_files = list(IMAGE_DIR.glob("*"))
    OUTPUT_DIR.mkdir(exist_ok=True)

    if len(image_files) == 0:
        print("No images found in test_images folder")
        return

    for path in image_files:
        image = cv2.imread(str(path))

        if image is None:
            print(f"{path.name}: image could not be loaded")
            continue

        result = contains_red_circle(image, path.name)

        if result:
            print(f"{path.name}: RED CIRCLE FOUND")
            cv2.imwrite(str(OUTPUT_DIR / path.name), image)
        else:
            print(f"{path.name}: FAILED detection (see DEBUG details above)")

        cv2.imshow("Image", image)
        cv2.imshow("Result", image)
        cv2.waitKey(10)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


##Notes:
# - Slight distortion may cause circularity to fall below the threshold, so if the detection fails 
# because it is filtering out distorted circles, we can try lowering the circularity threshold.

# - If the red circles are partially irregular, the fill ratio can drop below 0.7, 
#so we can try lowering the fill ratio threshold to allow for more partially obscured circles to be detected.

# - If the red circles are darker or lighter than expected, we can adjust the HSV color thresholds 
#to better capture the range of red in the images.

# - If the circle is smaller than expected, we can lower the minimum area threshold 
#to allow for smaller circles to be detected.

# - If there is an imperfect masking, it means the fill ratio is too strict, 
# so we can try lowering that.,



#Right now we just print area, circularity, and fill ratio to see why it failed or passed.
