import cv2

for i in range(5):
    # Read the original image
    path = "/home/ved2311/foldsformer/data/demo/CornersEdgesInward/rgbviz/"
    image = cv2.imread(path + str(i) + ".png")
    # Rotate the image by 90 degrees counterclockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save the rotated image
    cv2.imwrite(path + 'rotated' + "-" + str(i) + ".png", rotated_image)
