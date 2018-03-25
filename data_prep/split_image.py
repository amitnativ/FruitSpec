import numpy as np


def split_image(img, xsteps, ysteps, phase = 0):
    """

    :param img: large image
    :param xsteps: step size to split in x
    :param ysteps: step size to split in y
    :param phase: the area of overlap between images
    :return: small images from split image
    """

    (rows, cols, depth) = np.shape(img)
    cols = list(range(cols))
    rows = list(range(rows))

    xTopLeft = cols[0:-1:xsteps] + cols[phase:-1:xsteps]
    yTopLeft = rows[0:-1:ysteps] + rows[phase:-1:ysteps]

    xBottomRight = cols[xsteps:-1:xsteps] + cols[phase+xsteps:-1:xsteps]
    yBottomRight = rows[ysteps:-1:ysteps] + rows[phase+ysteps:-1:ysteps]

    while len(xBottomRight) < len(xTopLeft):
        xBottomRight.append(cols[-1])
    while len(yBottomRight) < len(yTopLeft):
        yBottomRight.append(rows[-1])

    xBottomRight[-1] = np.shape(img)[1]
    yBottomRight[-1] = np.shape(img)[0]

    img_split = []
    counter = 0
    for rind, y0 in enumerate(yTopLeft):
        img_split_row = []
        for cind, x0 in enumerate(xTopLeft):
            img_split_row.append(img[yTopLeft[rind]:yBottomRight[rind], xTopLeft[cind]:xBottomRight[cind]])
        img_split.append(img_split_row)

    # img = cv2.resize(img, None, fx = 0.2, fy = 0.2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return img_split, xTopLeft, yTopLeft, xBottomRight, yBottomRight