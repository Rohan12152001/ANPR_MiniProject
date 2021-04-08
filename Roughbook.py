
# Main
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


def fetchFromDB(number):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='alpr',
                                             user='root',
                                             password=str(os.environ.get('Pass')))
        cursor = connection.cursor(dictionary=True)
        sql_fetch_query = """select * from license_plates where PlateNumber=%s"""
        cursor.execute(sql_fetch_query, (number,))
        records = cursor.fetchone()
        # print(records)
        # print(len(records))
        if(len(records)>0):
            print(f'Number verified :) ')
        else:
            print("Number NOT verified :( ")

    except Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
        return


if __name__ == "__main__":

    # To take image using gui
    root = Tk()
    root.title("MINI PROJECT")

    string = filedialog.askopenfilename(initialdir="D:/code/ALPR",
                                        title="Select A File",
                                        filetypes=(("*.png", "*.jpg"), ("all files", "*.*")))

    # load image as an object
    img = cv.imread(string)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
    cv.imshow('Frame3', gray)
    cv.waitKey(2000)

    # basically bilateralFilter(img, d, sigmacolor, sigmaSpace, borderType)
    bfilter = cv.bilateralFilter(gray, 15, 15, 15) #Noise reduction
    cv.imshow('Frame3', bfilter)
    cv.waitKey(2000)

    # canny uses hystersis thresholding
    edged = cv.Canny(bfilter, 30, 250) #Edge detection
    # edged = cv.Canny(gray, 30, 200)

    #plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))
    cv.imshow('Frame3', edged)
    cv.waitKey(2000)

    keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    # img1 = cv.drawContours(edged, [contours[8]], 0, 255, -1)
    # cv.imshow('Frame3', img1)
    # cv.waitKey(6000)

    print("Contour length:", len(contours))

    location = None
    # for contour in contours:
    #   approx = cv.approxPolyDP(contour, 10, True)   # (cont, resolution, closed/open)
    #   if len(approx) == 4:
    #       location = approx
    #       break

    for contour in contours:
      approx = cv.approxPolyDP(contour, 10, True)
      """Can add condition for rectangle (basis of length & breadth) IMP """
      if len(approx) == 4 and 500 < cv.contourArea(contour):
          location = approx
          img1 = cv.drawContours(edged, [contour], 0, 255, -1)
          cv.imshow('Frame3', img1)
          cv.waitKey(2000)
          break

    # for contour in contours:
    #   approx = cv.approxPolyDP(contour, 10, True)
    #   """Can add condition for rectangle (basis of length & breadth) IMP """
    #   if len(approx) == 4:
    #       location = approx
    #       img1 = cv.drawContours(edged, [contour], 0, 255, -1)
    #       cv.imshow('Frame3', img1)
    #       cv.waitKey(2000)


    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv.drawContours(mask, [location], 0, 255, -1)
    cv.imshow('Frame3', new_image)
    cv.waitKey(2000)

    new_image = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('Frame3', new_image)
    cv.waitKey(2000)


    #plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    cv.imshow('Frame3', cropped_image)
    cv.waitKey(5000)

    #plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))


    # read text using easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped_image)


    # To solve the "IND" problem
    array = []
    for ele in result:
        if len(ele[-2]) > 7:
            array.append(ele[-2])
            break

    # results with the amount of confidence
    # print(result)

    text = array[0]

    # Strip off the spaces present in between characters
    text = text.replace(" ", "")

    # Check with the DB
    fetchFromDB(text)

    # text = result[0][-2]
    font = cv.FONT_HERSHEY_SIMPLEX
    res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60),
                   fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2,
                   lineType=cv.LINE_AA)
    res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    #plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))

    # show the frame with number
    cv.imshow('Frame3', res)
    cv.waitKey(7000)

    print("Text: ", text)