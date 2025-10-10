import numpy as np
import cv2

def getMiddlePoint(ids, avatar):
    points = []
    for id in ids:
        points.append(avatar.getPoint(id).toPoint())
    sumX = 0
    sumY = 0
    for point in points:
        sumX += point[0]
        sumY += point[1]

    return (int(sumX/len(points)), int(sumY/len(points)))
def getMiddlePointExt(xIds, yIds, avatar):
    xPoints = []
    yPoints = []
    for id in xIds:
        xPoints.append(avatar.getPoint(id).toPoint())
    for id in yIds:
        yPoints.append(avatar.getPoint(id).toPoint())
    sumX = 0
    sumY = 0
    for point in xPoints:
        sumX += point[0]
    for point in yPoints:
        sumY += point[1]

    return (int(sumX/len(xPoints)), int(sumY/len(yPoints)))
def getPointDistance(id1, id2, avatar):
    point1 = avatar.getPoint(id1).toPoint()
    point2 = avatar.getPoint(id2).toPoint()
    return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)


def drawOlho(idLeft, idRight, idUp, idDown, avatar, img):
    addx, addy = 200, 100

    olhoRadiusMult = 2
    blinkLineThickness = 12
    blinkBound = .25

    olhoSize = abs(avatar.getPoint(idDown).avatarY - avatar.getPoint(idUp).avatarY)
    olhoRatio = getPointDistance(idDown, idUp, avatar) / getPointDistance(idRight, idLeft, avatar)
    if (olhoRatio) > blinkBound:
        (x, y) = getMiddlePointExt([idRight, idLeft], [idUp, idDown], avatar)
        cv2.circle(img, (x+addx, y+addy), 0, (0,0,0), olhoSize*olhoRadiusMult)
    else:
        (xl, yl) = avatar.getPoint(idLeft).toPoint()
        (xr, yr) = avatar.getPoint(idRight).toPoint()
        cv2.line(img, (xl + addx, yl + addy), (xr + addx, yr + addy), (0,0,0), blinkLineThickness)  


def drawBoca(idLeft, idRight, idUp, idDown, avatar, img):
    addx, addy = 100, 100
    bocaRadiusMult = .01
    speakLineThickness = 36
    speakBound = .55
    smileBound = 350
    bocaW = getPointDistance(idRight, idLeft, avatar)
    bocaSize = abs(avatar.getPoint(idDown).avatarY - avatar.getPoint(idUp).avatarY)
    bocaRatio = getPointDistance(idDown, idUp, avatar) / bocaW
    print(f"boca ratio {bocaRatio}")
    if bocaW > smileBound:
        cv2.line(img, tuple(p+addx for p in avatar.getPoint(idLeft).toPoint()), tuple(p+addx for p in avatar.getPoint(idRight).toPoint()), (0,0,0), speakLineThickness*3) 
    elif (bocaRatio) > speakBound:
        cv2.circle(img, tuple(p+addx for p in getMiddlePointExt([idRight, idLeft], [idUp, idDown], avatar)), 0, (0,0,0), int((bocaSize**1.9)*bocaRadiusMult))
    else:
        cv2.line(img, tuple(p+addx for p in avatar.getPoint(idLeft).toPoint()), tuple(p+addx for p in avatar.getPoint(idRight).toPoint()), (0,0,0), speakLineThickness) 
    