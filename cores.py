import cv2
import matplotlib.pyplot as plt
import numpy as np

imagem_bgr = cv2.imread('imagem_original_frutas_cores.jpg')
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

print(f'tipo de dado: {imagem_rgb.dtype}')
print(f'Altura , Largura e canais: {imagem_rgb.shape}')
print(f'Total de pixels: {imagem_rgb.size}')

plt.title('Imagem original RGB')
plt.imshow(imagem_rgb)
plt.show()

imagem_vermelha = imagem_rgb[:,:,0]
imagem_verde = imagem_rgb[:,:,1]
imagem_azul = imagem_rgb[:,:,2]

plt.subplot(2,2,1)
plt.title('Imagem orginal RGB')
plt.imshow(imagem_rgb)
plt.subplot(2,2,2)
plt.title('Imagem vermelha')
plt.imshow(imagem_vermelha, cmap='gray')
plt.subplot(2,2,3)
plt.title('Imagem verde')
plt.imshow(imagem_verde, cmap='gray')
plt.subplot(2,2,4)
plt.title('Imagem azul')
plt.imshow(imagem_azul, cmap='gray')
plt.subplots_adjust(hspace=0.6)
plt.show()

pixel1 = imagem_rgb[89,130]
pixel2 = imagem_rgb[217,715]
pixel3 = imagem_rgb[405,129]

print(f"Pixel 1: {pixel1}")
print(f"Pixel 2: {pixel2}")
print(f"Pixel 3: {pixel3}")

histograma_vermelho = cv2.calcHist([imagem_rgb],[0],None,histSize=[256],ranges=[0,256])
histograma_verde = cv2.calcHist([imagem_rgb],[1],None,histSize=[256],ranges=[0,256])
histograma_azul = cv2.calcHist([imagem_rgb],[2],None,histSize=[256],ranges=[0,256])


plt.figure(figsize=(14,7))
plt.subplot(1,3,1)
plt.plot(histograma_vermelho)
plt.title('Canal vermelho')
plt.subplot(1,3,2)
plt.plot(histograma_verde)
plt.title('Canal verde')
plt.subplot(1,3,3)
plt.plot(histograma_azul)
plt.title('Canal azul')
plt.show()

img_hsv = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2HSV)
#plt.imshow(img_hsv)
#plt.show()
matiz_h = img_hsv[:,:,0]
saturation_s = img_hsv[:,:,1]
brilho_v = img_hsv[:,:,2]

#mantive uint8 entao 0 até 255
plt.figure(figsize=(14,7))
plt.subplot(1,4,1)
plt.imshow(matiz_h, cmap='gray')
plt.title('Matiz - H')
plt.subplot(1,4,2)
plt.imshow(saturation_s, cmap='gray')
plt.title('Saturação - S')
plt.subplot(1,4,3)
plt.imshow(brilho_v, cmap='gray')
plt.title('Brilho - V')
plt.subplot(1,4,4)
plt.imshow(imagem_rgb)
plt.title('Imagem original rgb')
plt.show()

pixel1_hsv = img_hsv[89,130]
pixel2_hsv = img_hsv[217,715]
pixel3_hsv = img_hsv[405,129]
print(f"Pixel 1: {pixel1_hsv}")
print(f"Pixel 2: {pixel2_hsv}")
print(f"Pixel 3: {pixel3_hsv}")

imagem_rgb_float32 = imagem_rgb.astype(np.float32)
print(imagem_rgb_float32.dtype)

R = imagem_rgb_float32[:,:,0]
G = imagem_rgb_float32[:,:,1]
B = imagem_rgb_float32[:,:,2]
Y = R+G+B

#coordenadas de cromaticidade -> normalização da cor 'eliminar sombra e iluminação da cena
Y_sem_zero = np.where(Y==0, 0.000001, Y)
r = R/Y_sem_zero
g = G/Y_sem_zero
b = B/Y_sem_zero
print(r+g+b)

plt.figure(figsize=(14,7))
plt.subplot(1,3,1)
plt.imshow(r, cmap='gray')
plt.title('r')
plt.subplot(1,3,2)
plt.imshow(g, cmap='gray')
plt.title('g')
plt.subplot(1,3,3)
plt.imshow(b, cmap='gray')
plt.title('b')
plt.show()
print()
print()
pixel1_r = r[89,130]
pixel2_r = r[217,715]
pixel3_r = r[405,129]
print(f"Pixel 1 - r: {pixel1_r}")
print(f"Pixel 2 - r: {pixel2_r}")
print(f"Pixel 3 - r: {pixel3_r}")

print()
pixel1_g = g[89,130]
pixel2_g = g[217,715]
pixel3_g = g[405,129]
print(f"Pixel 1 - g: {pixel1_g}")
print(f"Pixel 2 - g: {pixel2_g}")
print(f"Pixel 3 - g: {pixel3_g}")

print()
pixel1_b = b[89,130]
pixel2_b = b[217,715]
pixel3_b = b[405,129]
print(f"Pixel 1 - b: {pixel1_b}")
print(f"Pixel 2 - b: {pixel2_b}")
print(f"Pixel 3 - b: {pixel3_b}")
print()
print()
print(f'soma do pixel 1: {pixel1_g+pixel1_b+pixel1_r}')
print(f'soma do pixel 2: {pixel2_g+pixel2_b+pixel2_r}')
print(f'soma do pixel 3: {pixel3_g+pixel3_b+pixel3_r}')
pixel1_hsv = img_hsv[89,130]
pixel2_hsv = img_hsv[217,715]
pixel3_hsv = img_hsv[405,129]
print(f"Pixel 1- hsv: {pixel1_hsv}")
print(f"Pixel 2- hsv: {pixel2_hsv}")
print(f"Pixel 3- hsv: {pixel3_hsv}")
print()
pixel1 = imagem_rgb[89,130]
pixel2 = imagem_rgb[217,715]
pixel3 = imagem_rgb[405,129]

print(f"Pixel 1- RGB: {pixel1}")
print(f"Pixel 2- RGB: {pixel2}")
print(f"Pixel 3- RGB: {pixel3}")

#plotar o plano de cromaticidade
r_amostra = r[::20, ::20].flatten()
g_amostra = g[::20, ::20].flatten() #flatten levar 2d pra 1d pra scatter

plt.scatter(r_amostra,g_amostra,s=5,color='red',alpha=0.8, label= 'pixels da imagem')
plt.scatter(0.333, 0.333, s = 70, marker='x',color= 'blue', alpha=1.0, label='branco ideal')
plt.title('plano de cromaticidade')
plt.legend()
plt.show()

imagem_I2 = imagem_rgb_float32 * 0.5
imagem_I2_plotar = imagem_I2.astype(np.uint8)

plt.subplot(1,2,1)
plt.title('imagem original')
plt.imshow(imagem_rgb)
plt.subplot(1,2,2)
plt.title('imagem multiplicada por k=0.5')
plt.imshow(imagem_I2_plotar)
plt.show()

pixel1_reduzido = imagem_I2_plotar[89,130]
pixel2_reduzido = imagem_I2_plotar[217,715]
pixel3_reduzido = imagem_I2_plotar[405,129]

print(f"Pixel 1- RGB brilho reduzido: {pixel1_reduzido}")
print(f"Pixel 2- RGB brilho reduzido: {pixel2_reduzido}")
print(f"Pixel 3- RGB brilho reduzido: {pixel3_reduzido}")

R_brilho_reduzido= imagem_I2[:,:,0]
G_brilho_reduzido = imagem_I2[:,:,1]
B_brilho_reduzido = imagem_I2[:,:,2]
Y_brilho_reduzido = R_brilho_reduzido+G_brilho_reduzido+B_brilho_reduzido

#coordenadas de cromaticidade -> normalização da cor 'eliminar sombra e iluminação da cena
Y_sem_zero_brilho_reduzido = np.where(Y_brilho_reduzido==0, 0.000001, Y_brilho_reduzido)
r_brilho_reduzido = R_brilho_reduzido/Y_sem_zero_brilho_reduzido
g_brilho_reduzido = G_brilho_reduzido/Y_sem_zero_brilho_reduzido
b_brilho_reduzido = B_brilho_reduzido/Y_sem_zero_brilho_reduzido
print(r_brilho_reduzido+g_brilho_reduzido+b_brilho_reduzido)

print()
print()
pixel1_r_brilho_reduzido = r_brilho_reduzido[89,130]
pixel2_r_brilho_reduzido = r_brilho_reduzido[217,715]
pixel3_r_brilho_reduzido = r_brilho_reduzido[405,129]
print(f"Pixel 1 - r: {pixel1_r_brilho_reduzido}")
print(f"Pixel 2 - r: {pixel2_r_brilho_reduzido}")
print(f"Pixel 3 - r: {pixel3_r_brilho_reduzido}")

print()
pixel1_g_brilho_reduzido = g_brilho_reduzido[89,130]
pixel2_g_brilho_reduzido = g_brilho_reduzido[217,715]
pixel3_g_brilho_reduzido = g_brilho_reduzido[405,129]
print(f"Pixel 1 - g: {pixel1_g_brilho_reduzido}")
print(f"Pixel 2 - g: {pixel2_g_brilho_reduzido}")
print(f"Pixel 3 - g: {pixel3_g_brilho_reduzido}")

print()
pixel1_b_brilho_reduzido = b_brilho_reduzido[89,130]
pixel2_b_brilho_reduzido = b_brilho_reduzido[217,715]
pixel3_b_brilho_reduzido = b_brilho_reduzido[405,129]
print(f"Pixel 1 - b: {pixel1_b_brilho_reduzido}")
print(f"Pixel 2 - b: {pixel2_b_brilho_reduzido}")
print(f"Pixel 3 - b: {pixel3_b_brilho_reduzido}")
print()
print()
#mesmo caindo pela metade os canais, a cromaticidade permanece invariavel o mesmo valor

r_amostra_brilho_reduzido = r_brilho_reduzido[::20, ::20].flatten()
g_amostra_brilho_reduzido = g_brilho_reduzido[::20, ::20].flatten() #flatten levar 2d pra 1d pra scatter

plt.subplot(1,2,1)
plt.scatter(r_amostra,g_amostra,s=5,color='red',alpha=0.8, label= 'pixels da imagem')
plt.title('plano de cromaticidade')
plt.subplot(1,2,2)
plt.scatter(r_amostra_brilho_reduzido,g_amostra_brilho_reduzido,s=5,color='red',alpha=0.8, label= 'pixels da imagem brilho reduzido')
plt.title('plano de cromaticidade brilho reduzido')
plt.legend()
plt.show() 
#comprovada invariancia a intensidade de iluminação

#figura final
plt.figure(figsize=(20,10))
plt.subplot(3,3,1)
plt.title('imagem original')
plt.imshow(imagem_rgb)
plt.subplot(3,3,2)
plt.title('Canal R')
plt.imshow(imagem_vermelha, cmap='gray')
plt.subplot(3,3,3)
plt.title('Canal G')
plt.imshow(imagem_verde, cmap='gray')
plt.subplot(3,3,4)
plt.title('Canal B')
plt.imshow(imagem_azul, cmap='gray')
plt.subplot(3,3,5)
plt.title('H')
plt.imshow(matiz_h, cmap='gray')
plt.subplot(3,3,6)
plt.title('S')
plt.imshow(saturation_s, cmap='gray')
plt.subplot(3,3,7)
plt.title('V')
plt.imshow(brilho_v, cmap='gray')
plt.subplot(3,3,8)
plt.title('r')
plt.imshow(r, cmap='gray')
plt.subplot(3,3,9)
plt.title('g')
plt.imshow(g, cmap='gray')
plt.subplots_adjust(hspace=0.7)
plt.show()