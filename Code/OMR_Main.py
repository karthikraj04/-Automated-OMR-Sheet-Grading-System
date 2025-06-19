import cv2
import numpy as np
import utlis


pathImage = "1.jpg"
heightImg = 700
widthImg  = 700   
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]


img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))  
imgFinal = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  
imgCanny = cv2.Canny(imgBlur, 10, 70) 

# FIND ALL CONTOURS
imgContours = img.copy()
imgBigContour = img.copy()
contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# FILTER RECTANGLE CONTOURS
rectCon = utlis.rectContour(contours)

if len(rectCon) < 2:
    print("Error: Not enough rectangular contours found.")
    cv2.imshow("Error", img)
    cv2.waitKey(0)
    exit()

biggestPoints = utlis.getCornerPoints(rectCon[0]) 
gradePoints = utlis.getCornerPoints(rectCon[1])  
if biggestPoints.size != 0 and gradePoints.size != 0:

    # BIGGEST RECTANGLE WARPING
    biggestPoints = utlis.reorder(biggestPoints)
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
    pts1 = np.float32(biggestPoints)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # SECOND BIGGEST RECTANGLE WARPING
    gradePoints = utlis.reorder(gradePoints)
    cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
    ptsG1 = np.float32(gradePoints)
    ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utlis.splitBoxes(imgThresh)
    myPixelVal = np.zeros((questions, choices))

    countR = 0
    countC = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == choices:
            countC = 0
            countR += 1

    # FIND USER ANSWERS
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])

    # GRADING
    grading = [1 if ans[x] == myIndex[x] else 0 for x in range(0, questions)]
    score = (sum(grading) / questions) * 100

    # DISPLAY ANSWERS
    utlis.showAnswers(imgWarpColored, myIndex, grading, ans)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    utlis.showAnswers(imgRawDrawing, myIndex, grading, ans)

    # INVERSE TRANSFORMATION
    invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    # DISPLAY GRADE
    imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
    cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

    # FINAL IMAGE
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)


cv2.imshow("Final Result", imgFinal)
cv2.waitKey(0)

