# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:46:49 2024

@author: User
"""

# Imports necessários
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from PIL import Image, ImageDraw
from gtts import gTTS  # Biblioteca para conversão de texto em fala
import os  # Necessário para tocar o áudio gerado
import time

# Função para desenhar caixas de detecção na imagem e falar o nome dos objetos
def draw_boxes_and_speak(image, boxes, class_names, scores, max_boxes=10):
    """
    Função para desenhar as caixas de detecção de objetos na imagem
    e falar o nome dos objetos detectados.
    
    image: imagem a ser desenhada
    boxes: coordenadas das caixas delimitadoras
    class_names: nomes das classes dos objetos detectados
    scores: pontuações (confiabilidade) das detecções
    max_boxes: número máximo de caixas a desenhar
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= 0.7:  # Verificar se a pontuação é maior ou igual a 70%
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Definir a cor verde para as caixas com alta confiança
            color = "green"
            
            # Desenhar a caixa e o texto
            draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
            object_name = class_names[i].decode('utf-8')
            confidence = f"{int(scores[i] * 100)}%"
            draw.text((left, top), f"{object_name}: {confidence}", fill=color)
            
            # Falar o nome do objeto detectado
            tts = gTTS(text=object_name, lang='en')
            tts.save("detected_object.mp3")
            os.system("start detected_object.mp3")  # Tocar o áudio (Windows)
            time.sleep(2)  # Aguardar para dar tempo de tocar o áudio

# Função para realizar o reconhecimento de objetos
def detect_objects(model, image_np):
    """
    Função para realizar a detecção de objetos em uma imagem.
    
    model: modelo carregado do TensorFlow Hub
    image_np: imagem em formato numpy
    """
    # Expandir a imagem para corresponder ao formato esperado pelo modelo
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Converter a imagem para float32
    converted_img = tf.image.convert_image_dtype(image_np_expanded, dtype=tf.float32)
    
    # Acessar a função de assinatura específica para inferência
    detector = model.signatures.get('default')
    if detector is None:
        print("Erro: a assinatura 'default' não está disponível no modelo.")
        return None
    
    # Tentar rodar a inferência com o modelo
    try:
        result = detector(converted_img)
    except Exception as e:
        print(f"Erro ao realizar a inferência: {e}")
        return None
    
    # Converter o resultado em um dicionário legível
    result = {key: value.numpy() for key, value in result.items()}
    return result

# Função principal para capturar a imagem da câmera e realizar a detecção
def run_object_detection():
    # Carregar o modelo do TensorFlow Hub (Inception ResNet V2)
    model_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    model = hub.load(model_url)
    
    # Iniciar a captura da câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera!")
        return

    print("Pressione 'q' para sair.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar o frame da câmera.")
                break

            # Converter a imagem para formato compatível com o modelo
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_np)

            # Realizar a detecção de objetos
            start_time = time.time()
            result = detect_objects(model, image_np)
            end_time = time.time()

            if result is None:
                print("Erro ao detectar objetos. Interrompendo...")
                break

            # Desenhar as caixas de detecção na imagem e falar o nome dos objetos
            draw_boxes_and_speak(pil_image, result['detection_boxes'], result['detection_class_entities'], result['detection_scores'])

            # Mostrar a imagem com as detecções
            output_image = np.array(pil_image)
            cv2.imshow('Reconhecimento de Objetos', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

            print(f"Tempo de inferência: {end_time - start_time:.2f} segundos")

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Liberar a câmera e fechar as janelas
        cap.release()
        cv2.destroyAllWindows()

# Executar a função principal
if __name__ == "__main__":
    run_object_detection()
