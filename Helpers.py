# IMP !!
"""import cv2
import numpy as np

cap = cv2.VideoCapture('park2.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

# out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    cv.imshow("new", dilated)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 5000 or cv2.contourArea(contour) < 500:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    # out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
# out.release()"""


# For number extract from an image web !!
"""import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

img = cv2.imread('./Images/Ind2.PNG', cv2.IMREAD_COLOR)

# img = cv2.resize(img, (620, 480))
img = cv2.resize(img, (620, 480))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print
    "No contour detected"
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape, np.uint8)
# if(screenCnt==None):
#     print("None")
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
new_image = cv2.bitwise_and(img, img, mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Read the number plate
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Detected Number is:", text)

cv2.imshow('image', img)
cv2.imshow('Cropped', Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()"""

# For images
"""img = cv.imread('image1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))


bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))

keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]


location = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv.drawContours(mask, [location], 0,255, -1)
new_image = cv.bitwise_and(img, img, mask=mask)


plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))


reader = easyocr.Reader(['en'], gpu='cuda:1')
result = reader.readtext(cropped_image)


text = result[0][-2]
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))"""


"""kernal = np.ones((2, 2), np.uint8)
print(kernal)

img = cv.imread('Cars12.png', 0)

_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

cv.imshow("Image", img)
# cv.imshow("th1", th1)
cv.imshow("th2", th2)

cv.waitKey(0)
cv.destroyAllWindows()"""

"""img = cv.imread('Cars12.png', cv.IMREAD_GRAYSCALE)
lap = cv.Laplacian(img, cv.CV_64F, ksize= 3)
lap = np.uint8(np.absolute(lap))

cv.imshow("Image", lap)
# cv.imshow("th1", th1)

cv.waitKey(0)
cv.destroyAllWindows()"""

"""img = cv.imread('Cars12.png', cv.IMREAD_GRAYSCALE)
lap = cv.Canny(img,100,200)
# lap = np.uint8(np.absolute(lap))

cv.imshow("Image", lap)
# cv.imshow("th1", th1)

cv.waitKey(0)
cv.destroyAllWindows()"""

"""img = cv.imread('Trial.PNG')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hirearchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv.drawContours(img, contours, -1, (0, 255, 0), 2)

cv.imshow("Image", img)
cv.imshow("Image Gray", imgray)

cv.waitKey(0)
cv.destroyAllWindows()"""