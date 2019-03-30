
import numpy as np
import threading
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from concurrent.futures import ProcessPoolExecutor

class ClearDeadSpaceThread(threading.Thread):

    def __init__(self, threadId, cap, bgSubtractor, iterations):

        threading.Thread.__init__(self)
        self.threadId = threadId
        self.cap = cap
        self.iterations = iterations
        self.bgSubtractor = bgSubtractor
        self.processedFrames = []

    def run(self):

        i = 0
        windowName = "thread" + str(self.threadId)
        cv2.namedWindow(windowName, 0)
        cv2.resizeWindow(windowName, 500, 500)


        while i < self.iterations and self.cap.isOpened():

            ret, frame = self.cap.read()

            if not ret:
                break

            fgmask = self.bgSubtractor.apply(frame)
            fgmask = self.filter_mask(fgmask)

            is_active, contours = self.is_in_motion(fgmask)


            copy = frame.copy()

            if is_active:
                self.processedFrames.append(frame)
                cv2.drawContours(copy, contours, -1, (0,255,0), 5)

            cv2.imshow(windowName, copy)
            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;


        self.cap.release()


    def is_in_motion(self, frame, min=10000):

        contours, heirarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        amountOfContours = len(contours)
        if amountOfContours == 0:
            return (False, None)

        for i in range(amountOfContours):
            contour = contours[i]
            area = cv2.contourArea(contour)
            # meterArea = area + cv2.arcLength(contour, True)

            # If any is minimum area, then there was something in motion.
            if (area >= min):
                return (True, contours)

        return (False, None)

    # This filters it to make objects in frame more accurate.
    def filter_mask(self,frame):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fills small hole
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        # Gets rid of white noise.
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Understand both blur.
        guasBlur = cv2.GaussianBlur(opening, (5, 5), 0)
        blur = cv2.blur(guasBlur, (5, 5))

        dilate = cv2.dilate(blur, kernel, iterations=5)

        # I see, now if less than 240, 0 otherwise 255
        ret, th = cv2.threshold(dilate, 175, 255, cv2.THRESH_BINARY)
        # th = dilation[dilation < 240] = 10
        return th

#Pretains model to recognize what is considered background.
def train_bg_subtractor(instance, cap, num = 500):


    i = 0
    #So this learns to differentiate between foreground and background.
    while i < num and cap.isOpened():
        ret,frame = cap.read()
        if ret:
           instance.apply(frame)

        i += 1


def findEvenSplit(frameCount, minimum):

    #If still even and over 600 frames, half again.
    while frameCount > minimum:
        frameCount /= 2
        #Sum of this number will be whole number.
    frameCount /= 2



    #Otherwise or when it starts returning, spawn the threads?
    return frameCount


def main():


    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)



    trainingCapture = cv2.VideoCapture('Porridge.avi')
    cap = cv2.VideoCapture('Porridge.avi')

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    train_bg_subtractor(fgbg, trainingCapture, 500)
    frames = []

    fourcc =  cv2.VideoWriter_fourcc('X','V','I','D')

    # input frames per second.
    out = cv2.VideoWriter('output.avi', fourcc, 60.0, (int(cap.get(3)), int(cap.get(4))))

    threads = []

    print("Frame count " + str(frameCount))

    evenSplit = findEvenSplit(frameCount, frameCount / 2)

    print("Even split " + str(evenSplit))
    i = 0

    #Since can't copy the stream, gotta make two for every single one.
    #Or, only train on one.
    while (i + evenSplit < frameCount):

        subCapture = cv2.VideoCapture('Porridge.avi')

        subCapture.set(cv2.CAP_PROP_POS_FRAMES,i)

        thread = ClearDeadSpaceThread(i, subCapture, fgbg, evenSplit)

        thread.start()
        #Hindsight could've done this with futures and the pool execetutor.
        threads.append(thread)



        i += evenSplit

    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    thread = ClearDeadSpaceThread(i, cap, fgbg, frameCount - i)
    thread.start()

    threads.append(thread)

    for thread in threads:

        #Wait for first thread to finish.
        thread.join()

        #Concatenate result
        for frame in thread.processedFrames:
            out.write(frame)

    out.release()
    cap.release()
    movieClip = VideoFileClip('output.avi')
    audioClip = AudioFileClip('Porridge.avi')
    movieClip = movieClip.set_audio(audioClip)
    movieClip.write_videofile('output_moviePy.avi', fps=60, codec='rawvideo')

    cv2.destroyAllWindows()



if __name__ ==  "__main__":
    main()




#Exports video clips.

