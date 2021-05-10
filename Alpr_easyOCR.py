
# Working code for video input !! (easyOCR)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
import easyocr
import time
from tkinter import filedialog
from tkinter import *
import mysql.connector, sys, os
from mysql.connector import Error

def fetchDB(number):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='alpr',
                                             user='root',
                                             password=str(os.environ.get('Pass')))
        cursor = connection.cursor(dictionary=True)
        sql_fetch_query = """select * from license_plates where PlateNumber=%s"""
        cursor.execute(sql_fetch_query, (number,))
        records = cursor.fetchone()
        #print(records)
        # print(len(records))
        if(len(records)>0):
            print(f'Number verified !')

    except Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
        return


def HelperFunc(string):
  # print(Frame)
  img = cv.imread(string)

  # gray scale
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))

  # basically bilateralFilter(img, d, sigmacolor, sigmaSpace, borderType)
  # considers spatial & photometric distance
  bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction

  # canny uses hystersis thresholding (two level threshold)
  edged = cv.Canny(bfilter, 30, 200) #Edge detection

  #plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))

  # Finding out contours  & Sort them
  keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

  # locating the number plate using the contours
  location = None
  for contour in contours:
      approx = cv.approxPolyDP(contour, 10, True)
      if len(approx) == 4:
          location = approx
          break

  mask = np.zeros(gray.shape, np.uint8)
  new_image = cv.drawContours(mask, [location], 0, 255, -1)
  new_image = cv.bitwise_and(img, img, mask=mask)

  # cv.imshow('Frame3', new_image)
  # cv.waitKey(3000)

  #plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))

  (x, y) = np.where(mask == 255)
  (x1, y1) = (np.min(x), np.min(y))
  (x2, y2) = (np.max(x), np.max(y))
  cropped_image = gray[x1:x2+1, y1:y2+1]

  #plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))

  # read text using easyocr
  reader = easyocr.Reader(['en'], gpu=False)
  result = reader.readtext(cropped_image)

  text = result[0][-2]
  font = cv.FONT_HERSHEY_SIMPLEX
  res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60),
                   fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2,
                   lineType=cv.LINE_AA)
  res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
  #plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))

  # show the frame with number
  cv.imshow('Frame2', res)
  print("Text: ", text)

  # Strip off the spaces present in between characters
  text = text.replace(" ", "")

  # check if number is in DB
  if len(text) == 10:
      fetchDB(text.upper())

  """# check if number is in DB ?
  if text.upper() == Number[0]:
      h, w, c = img.shape
      cv.putText(img, "Flag: 1", (20, int(h)-10), font, 1, (0, 255, 0), 3)"""

if __name__ == "__main__":
    # To take image using gui
    root = Tk()
    root.title("MINI PROJECT")

    string = filedialog.askopenfilename(initialdir="D:/code/ALPR",
                                        title="Select A File")

    # Read the video from specified path
    cam = cv.VideoCapture(string)
    # cam = cv.VideoCapture(0)

    # time in epoch
    currentTime = int(time.time())

    while (cam.isOpened()):
        # reading from frame
        ret, frame = cam.read()
        cv.imshow('Frame', frame)

        if ret:
            currentTime2 = int(time.time())
            if(currentTime2!=currentTime and currentTime2 % 2 == 0):
                cv.imwrite("NewPicture.jpg", frame)
                try:
                  start = time.perf_counter()
                  HelperFunc("NewPicture.jpg")
                  finish = time.perf_counter()
                  print(f'Finish in {round(finish - start, 2)} seconds')
                except:
                  print("Error")
                finally:
                  currentTime = currentTime2
        # else:
        #     break
        if cv.waitKey(25) and 0xFF == ord('q'):
            break

    # Release all space and windows once done
    cam.release()
    cv.destroyAllWindows()



