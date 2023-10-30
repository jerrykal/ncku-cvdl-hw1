import cv2
import numpy as np


def stereo_disparity_map(img_l_path, img_r_path):
    """
    Compute disparity map from stereo images.
    And draw correspondence on the right image when clicked on the left image.
    """
    image_l = cv2.imread(img_l_path)
    image_r = cv2.imread(img_r_path)

    # Compute disparity map
    stereo = cv2.StereoBM.create(numDisparities=256, blockSize=25)
    disparity = stereo.compute(
        cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY),
    )
    disparity = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F  # type: ignore
    )

    # Draw correspondence on the right image
    cv2.namedWindow("imgL")
    param = [image_r, disparity]
    cv2.setMouseCallback("imgL", draw_correspondence, param)

    # Show images
    cv2.imshow("imgL", image_l)
    cv2.imshow("imgR", image_r)
    cv2.imshow("disparity", disparity.astype("uint8"))

    # Wait for ESC key to exit
    while True:
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def draw_correspondence(event, x, y, flags, param):
    """Draw correspondence on the right image."""
    if event == cv2.EVENT_LBUTTONDOWN:
        image_r, disparity = param
        if disparity[y, x] == 0:
            print("Failure case")
        else:
            tmp = image_r.copy()
            cv2.circle(tmp, (x - int(disparity[y, x]), y), 10, (0, 255, 0), -1)
            cv2.imshow("imgR", tmp)
            print(f"({x}, {y}),dis:{disparity[y, x]:.3f}")
