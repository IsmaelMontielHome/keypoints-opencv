import cv2

# Cargar la imagen
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear el detector ORB
orb = cv2.ORB_create()

# Encontrar los keypoints con ORB
keypoints = orb.detect(gray, None)

# Crear el descriptor BRIEF
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Calcular los descriptores BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)

# Dibujar los keypoints en la imagen
img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,5), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Mostrar la imagen con keypoints
cv2.imshow('Imagen con Keypoints BRIEF', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('keypoint_brief.jpeg', img_with_keypoints)
