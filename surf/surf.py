import cv2

# Cargar la imagen
img = cv2.imread("img.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Crear el objeto AKAZE y detectar y computar los puntos clave y descriptores
surf = cv2.AKAZE_create()
keypoints, descriptors = surf.detectAndCompute(imgGray, None)

# Dibujar los puntos clave en la imagen
imgGray_with_keypoints = cv2.drawKeypoints(imgGray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen en una ventana
cv2.imshow('Imagen con keypoints', imgGray_with_keypoints)
cv2.waitKey(0)  # Espera hasta que se presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas

# Guardar la imagen
cv2.imwrite('imagen_con_keypoints_surf.jpg', imgGray_with_keypoints)

#SE UTILIZO AKAZE YA QUE SURF FUE ELIMINIADO EN LA VERSION 4.0 DE OPENCV YA QUE NO SE LE HABIAN IMPLENTADO NUEVAS COSAS