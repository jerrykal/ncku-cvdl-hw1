import cv2


def draw_keypoints(imgpath):
    """Draws keypoints on an image and displays it."""
    if imgpath == "":
        return

    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors.
    sift = cv2.SIFT.create()
    kp, _ = sift.detectAndCompute(image, None)  # type: ignore

    # Draw keypoints on the image.
    outimage = cv2.drawKeypoints(image, kp, None, (0, 255, 0))  # type: ignore
    cv2.imshow("Keypoints", outimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_matched_keypoints(imgpath_1, imgpath_2):
    """Draws matched keypoints on two images and displays it."""
    if imgpath_1 == "" or imgpath_2 == "":
        return

    image_1 = cv2.imread(imgpath_1, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(imgpath_2, cv2.IMREAD_GRAYSCALE)

    # Detect keypoints and compute descriptors.
    sift = cv2.SIFT.create()
    kp_1, dec_1 = sift.detectAndCompute(image_1, None)  # type: ignore
    kp_2, dec_2 = sift.detectAndCompute(image_2, None)  # type: ignore

    # Match descriptors.
    good = []
    matches = cv2.BFMatcher().knnMatch(dec_1, dec_2, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Draw matched keypoints on the images.
    outimage = cv2.drawMatchesKnn(  # type: ignore
        image_1,
        kp_1,
        image_2,
        kp_2,
        good,
        None,  # type: ignore
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matched Keypoints", outimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
