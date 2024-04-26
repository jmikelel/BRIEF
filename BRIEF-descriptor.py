"""
TAREA (09)
VISION COMPUTACIONAL
PR 24
UDEM

ALUMNOS:
JESUS LEONARDO TAMEZ GLORIA
587400-IRSI-6TO

JOSE MIGUEL GONZALEZ ZARAGOZA
631145-IRSI-6TO

Demo using OpenCV for object detection with the image descriptor BRIEF

"""
import numpy as np
import cv2 as cv
import argparse as arg

def read_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE) #poner imagenes en escala de grises
    if img is None:
        print("Image Not Available")
        return None
    return img

def read_parser():
    """
    Parse user arguments.

    OUTPUT:
        argparse.ArgumentParser: Object containing parsed arguments. 
    """
    parser = arg.ArgumentParser(description="Program for feature detection and matching with BRIEF descriptor using STAR detector")
    parser.add_argument("--query_image", 
                        dest="query_path", 
                        type=str,  #En esta parte, usamos dos parser, pues es necesario para que el usuario pueda comparar dos imagenes
                        help="Path to query image")
    parser.add_argument("--train_image", 
                        dest="train_path", 
                        type=str, 
                        help="Path to train image")
    args = parser.parse_args()
    return args

def BRIEF_DESCRIPTOR(img):
    """
    Se buscan puntos de interes en la imagen, usando el detector STAR y el descriptor BRIEF

    INPUT: 
        img(np.ndarray): imagen de entrada determinada por el usuario
    OUTPUT:
        tuple: valores agrupados sobre los keypoints y descriptores, kp y des respectivamente
    """

    # Initiate STAR detector
    star = cv.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img, None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)

    print( brief.descriptorSize() )
    print( des.shape )

    return kp, des #Regresa los valores para keypoints y descriptors

def match_images(query_img, train_img): #Usando las imagenes de argparse proporcionadas por el usuario
    """
    Busca la relacion entre los descriptores de las imagenes de entrada y de entrenamiento

    Input data:
        query_img (np.ndarray): Imagen de entrada
        train_img (np.ndarray): Imagen para entrenamiento
    """
    
    # Find keypoints and descriptors for query image
    kp1, des1 = BRIEF_DESCRIPTOR(query_img)
    des1 = np.float32(des1)

    # Find keypoints and descriptors for train image
    kp2, des2 = BRIEF_DESCRIPTOR(train_img)
    des2 = np.float32(des2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN matcher
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Matching descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv.drawMatches(query_img, kp1, train_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#usamos drawMatches para mostrar los keypoints tomados entre ambas imagene,s luego usando flags, los hacemos notar 

    # Show matches
    cv.namedWindow("Matches", cv.WINDOW_NORMAL)  # Crear una ventana redimensionable
    cv.resizeWindow("Matches", 1000, 800)  # Establecer el tamaño deseado de la ventana
    cv.imshow("Matches", img_matches) #Muestra en la imagen matches las dos imagenes, donde se hace el match entre los kp de ambas
    cv.waitKey(0)

def main():
    """
    Función principal
    """
    args = read_parser()
    query_img = read_image(args.query_path)
    if query_img is None:           
        return

    train_img = read_image(args.train_path)
    if train_img is None:
        return

    # Coincidencia de imágenes
    match_images(query_img, train_img)

if __name__ == "__main__":
    main()