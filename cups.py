#!/usr/bin/env python2
from collections import deque
from functools import reduce
from functools import partial
import numpy as np
import imutils
import cv2
import os
import time

BUFFER = 16  # number of frames tracked
VIDEO_FILE = \
    os.path.abspath('resources' + os.sep + 'WP_20161120_17_43_25_Pro.mp4')


class Cups:
    __COLORS = [(0, 152, 255), (57, 220, 205), (183, 58, 103)]

    def __init__(self, buffer, file):
        self.__pts = [deque(maxlen=buffer) for i in range(0, 3)]
        self.__camera = cv2.VideoCapture(file)
        self.__switcheroo = None
        self.__missingPointNumbers = None

    def show(self):
        self.figureOutColor()
        self.findOtherFrames()

        self.__camera.release()
        cv2.destroyAllWindows()

    def figureOutColor(self):
        '''Finds lower and upper color bound determined by shapes found in the
        first frame.
        Sets __lowerBound and __upperBound which are hsv color boundaries for
        filtering'''
        grabbed, frame = self.__camera.read()

        image = imutils.resize(frame, width=600)  # resize
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # to check for color

        ''' We go with double dilate and double erosion instead of
        closing because it makes contours more visible
        Canny parameters are arbitrarily chosen, this is area for improvement.
        Moreover we dilate hard after canny filter to make the shapes whole
        We take biggest contours which take less than 20% of the image area.
        We find this the best way to determine our objects, because finding
        three colors similar to each other on a list of many is really
        impossible - there can always be three objects with similar colors that
        are something from the background'''
        contours = compose(
            lambda x: sorted(x, key=cv2.contourArea, reverse=True),
            lambda x: cv2.findContours(x.copy(), cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)[-2],
            lambda x: cv2.dilate(x, None, iterations=4),
            lambda x: cv2.Canny(x, 10, 180),
            lambda x: cv2.erode(x, None, iterations=2),
            lambda x: cv2.dilate(x, None, iterations=2),
            lambda x: cv2.GaussianBlur(x, (11, 11), 1)
        )(image)

        centersList = []
        for c in contours:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if cv2.contourArea(c) < (0.2*image.shape[0]*image.shape[1]):
                centersList.append((center, tuple(hsv[center[1]][center[0]])))

        colorsList = []
        for number, (center, color) in enumerate(centersList[:3]):
            self.__pts[number].append(center)
            colorsList.append(color)

        colorsList.sort()

        '''We set color bounds based found shapes.
        The critical parameter here is hue, since saturation proves to be 255
        for colored cups, which are usually the case.
        This means that is we used reflective cups they would be impossible to
        recognize'''
        self.__lowerBound = (max(colorsList[0][0] - 5, 0),
                             max(colorsList[0][1] - 20, 0),
                             max(colorsList[0][2] - 50, 0))

        self.__upperBound = (min(colorsList[2][0] + 10, 255),
                             min(colorsList[2][1] + 10, 255),
                             min(colorsList[2][2] + 50, 255))

    def findOtherFrames(self):
        '''Finds three points for each frame and adds them to queues which
        determine movement of our objects'''
        while (True):
            grabbed, frame = self.__camera.read()
            if not grabbed:
                break

            image = imutils.resize(frame, width=600)

            contours = compose(
                lambda x: filter(lambda y: cv2.contourArea(y) > 300, x),
                lambda x: sorted(x, key=cv2.contourArea, reverse=True),
                lambda x: cv2.findContours(x, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)[-2],
                lambda x: cv2.dilate(x, None, iterations=2),
                lambda x: cv2.erode(x, None, iterations=2),
                lambda x: cv2.inRange(x, self.__lowerBound, self.__upperBound),
                lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV),
                lambda x: cv2.GaussianBlur(x, (11, 11), 1)
            )(image)

            centers = []
            for i, c in enumerate(contours[:3]):
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                centers.append(center)

            self.appendPoints(centers)
            self.drawLines(image)

            cv2.imshow("Cups", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def appendPoints(self, centers):
        if len(centers) < 3:
            '''There are only 2 shapes, meaning two cups are on top of each
            other. We remember which are missing in __missingPointNumbers. We
            also set __switcheroo to remember that we need to swap them'''
            self.__switcheroo = True
            p = sorted([self.findClosest(point) for point in centers])
            missingPoint = sum(xrange(p[0], p[-1]+1)) - sum(p)
            self.__missingPointNumbers = (
                missingPoint,
                self.findClosest(self.__pts[missingPoint][0], closest=1)
            )
        elif self.__switcheroo == True:
            '''Now we need to swap points we remember are switching'''
            for point in centers:
                i = self.findClosest(point)
                self.__pts[i].append(point)
            first, second = self.__missingPointNumbers
            x = self.__pts[first].pop()
            y = self.__pts[second].pop()
            self.__pts[first].append(y)
            self.__pts[second].append(x)
            self.__switcheroo = False
            self.__missingPointNumbers = None
        else:
            '''All three cups are present'''
            for point in centers:
                i = self.findClosest(point)
                self.__pts[i].append(point)

    def findClosest(self, point, closest=0):
        '''Find which queue end is the closes to given point'''
        list = [(
            np.linalg.norm(np.array(point) - np.array(self.__pts[i][-1])), i)
            for i in range(0, len(self.__pts))
        ]

        list.sort(key=lambda x: x[0])
        return list[closest][1]

    def drawLines(self, frame):
        '''Takes each of the queues (paths) and draws a colored line for each
        of them.
        The line grows smaller with each point, so we can see which point is
        the newest'''
        for points, color in zip(self.__pts, self.__COLORS):
            for i in xrange(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue

                thickness = int(np.sqrt(60 / float(len(self.__pts[0]) - i)))
                cv2.line(frame, points[i - 1], points[i], color, thickness)


def compose(*functions):
    '''Small helper function for calling multiple one argument functions
    one after another. We used code from a comment found at:
    https://mathieularose.com/function-composition-in-python/#solution
    This is in our minds an elegant solution, which happens to fit perfectly in
    image processing where you constantly assign function result to a variable
    only to pass it to the next funciton.
    Please note that the functions passed as arguments are counter-intuitively
    in reversed order, since when you call h(g(f(x))) f is computed first, but
    h is called first, so h, g ,f are composed and not f, g, h'''
    return lambda x: reduce(lambda v, f: f(v), reversed(functions), x)

if __name__ == '__main__':
    instance = Cups(BUFFER, VIDEO_FILE)
    instance.show()
