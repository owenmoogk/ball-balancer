import cv2

img = cv2.imread('sample.jpg')
if img is None:
    raise FileNotFoundError("Image not found")

def show_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = img[y, x]
        print(f"RGB = ({r}, {g}, {b})")

cv2.namedWindow('image')
cv2.setMouseCallback('image', show_color)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cv2.destroyAllWindows()

