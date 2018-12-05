#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Histogram help: https://nrsyed.com/2018/02/08/real-time-video-histograms-with-opencv-and-python/
# Gamma help: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# This could be useful: https://gist.github.com/jwhendy/12bf558011fe5ff58bd5849954e84af4


MIRROR = True

hist_bins = 16
gamma_resolution = 20
contrast_resolution = 20


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1. / gamma if gamma > 0 else gamma_resolution
    table = np.array([((i / 255.) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    # One of these two could control exposure with certain cameras, first two are confusing
    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    #cam.set(cv2.CAP_PROP_EXPOSURE, 60)
    #subprocess.check_call("v4l2-ctl -d /dev/video0 -c exposure_absolute=40",shell=True)

    cv2.namedWindow('camera')
    cv2.createTrackbar('Gamma', 'camera', gamma_resolution, 3 * gamma_resolution, lambda x: None)
    cv2.createTrackbar('Brightness', 'camera', 50, 100, lambda x: None)
    cv2.createTrackbar('Contrast', 'camera', contrast_resolution, 3 * contrast_resolution, lambda x: None)

    ret_val, img = cam.read()
    num_pixels = np.prod(img.shape[:2])

    fig, ax = plt.subplots()
    lw = 3
    alpha = 0.5

    lineR, = ax.plot(np.arange(hist_bins), np.zeros((hist_bins,)), c='r', lw=lw, alpha=alpha)
    lineG, = ax.plot(np.arange(hist_bins), np.zeros((hist_bins,)), c='g', lw=lw, alpha=alpha)
    lineB, = ax.plot(np.arange(hist_bins), np.zeros((hist_bins,)), c='b', lw=lw, alpha=alpha)

    ax.set_xlim(0, hist_bins - 1)
    ax.set_ylim(0, 1)
    plt.ion()
    plt.show()

    while True:
        ret_val, img = cam.read()

        # Process the image
        gamma = cv2.getTrackbarPos('Gamma', 'camera') / gamma_resolution
        alpha = cv2.getTrackbarPos('Contrast', 'camera') / contrast_resolution
        beta = cv2.getTrackbarPos('Brightness', 'camera') - 50
        img = adjust_gamma(img, gamma)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        cv2.imshow('camera', cv2.flip(img, 1) if MIRROR else img)

        # Update the histogram
        (b, g, r) = cv2.split(img)
        histogramR = cv2.calcHist([r], [0], None, [hist_bins], [0, 255]) / num_pixels
        histogramG = cv2.calcHist([g], [0], None, [hist_bins], [0, 255]) / num_pixels
        histogramB = cv2.calcHist([b], [0], None, [hist_bins], [0, 255]) / num_pixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
        fig.canvas.draw()

        # Break on `Esc`
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
