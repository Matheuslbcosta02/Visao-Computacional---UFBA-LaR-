
import cv2
import matplotlib.pyplot as plt
import numpy as np

imagem_bgr = cv2.imread('imagem_momentos.jpg')
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(imagem_rgb)
plt.title('Imagem original')
plt.show()

# Converter para escala de cinza
cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

# Aplicar limiar manual
limiar, mascara = cv2.threshold(
    cinza,
    220,
    255,
    cv2.THRESH_BINARY_INV
)

print("Limiar utilizado:", limiar)

plt.imshow(mascara, cmap='gray')
plt.title('Máscara com limiar 220')
plt.show()

# Identificar os componentes conectados
num_labels, labels, stats, centroides = cv2.connectedComponentsWithStats(
    mascara,
    connectivity=8
)


for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    print(f"Componente {i}: {area} pixels")


areas = stats[1:, cv2.CC_STAT_AREA]
maior_componente = np.argmax(areas) + 1

mascara_objeto = (labels == maior_componente).astype(np.uint8)


area_objeto = np.sum(mascara_objeto)

print("Componente selecionado:", maior_componente)
print("Área do objeto:", area_objeto, "pixels")


plt.imshow(mascara_objeto, cmap='gray')
plt.title('Maior componente selecionado')
plt.show()

momento_zero = np.sum(mascara_objeto)
print(f'momento zero: {momento_zero}')

v_vertical, u_horizontal = np.indices(mascara_objeto.shape)

momento_10 = np.sum(u_horizontal * mascara_objeto)
momento_01 = np.sum(v_vertical * mascara_objeto)

u_centroide = momento_10/momento_zero
v_centroide = momento_01/momento_zero
print(f'Momento m10: {momento_10}')
print(f'Momento m01: {momento_01}')

print(f'Centroide X: {u_centroide:.2f}')
print(f'Centroide Y: {v_centroide:.2f}')
plt.figure(figsize=(10, 7))

plt.imshow(mascara_objeto, cmap='gray')

plt.scatter(
    u_centroide,
    v_centroide,
    color='red',
    marker='x',
    s=150,
    label='Centroide'
)

plt.title('Centroide do objeto')
plt.xlabel('Coordenada X (u)')
plt.ylabel('Coordenada Y (v)')
plt.legend()
plt.show()

mi_20 = np.sum(((u_horizontal - u_centroide)**2)*mascara_objeto)
mi_02 = np.sum(((v_vertical - v_centroide)**2)*mascara_objeto)
mi_11 = np.sum(((u_horizontal - u_centroide)*(v_vertical-v_centroide))*mascara_objeto)
matriz_J = np.array([
    [mi_20, mi_11],
    [mi_11, mi_02]
])

print("Matriz de segunda ordem:")
print(matriz_J)

# Calcular autovalores e autovetores
autovalores, autovetores = np.linalg.eig(matriz_J)

# Ordenar os autovalores do maior para o menor
ordem = np.argsort(autovalores)[::-1]

autovalores = autovalores[ordem]
autovetores = autovetores[:, ordem]

# Identificar o eixo principal e o secundário
autovalor_principal = autovalores[0]
autovalor_secundario = autovalores[1]

autovetor_principal = autovetores[:, 0]
autovetor_secundario = autovetores[:, 1]

# Apresentar os resultados
print("\n--- ETAPA 7: AUTOVALORES E AUTOVETORES ---")

print("Autovalores ordenados:")
print(autovalores)

print("\nAutovetor principal:")
print(autovetor_principal)

print("\nAutovetor secundário:")
print(autovetor_secundario)

# Determinar a orientação principal
q_u = autovetor_principal[0].real
q_v = autovetor_principal[1].real

theta = np.arctan2(q_v, q_u)

# Converter de radianos para graus
theta_graus = np.degrees(theta)

print(f'Orientação principal: {theta_graus:.2f} graus')

from matplotlib.patches import Ellipse
from math import sqrt

a = 2* sqrt(autovalor_principal.real/momento_zero)
b = 2* sqrt(autovalor_secundario.real/momento_zero)




elipse = Ellipse(
    (u_centroide, v_centroide),
    width=2*a,
    height=2*b,
    angle=theta_graus,
    fill=False,
    color='red'
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(mascara_objeto, cmap='gray')
ax.add_patch(elipse)
ax.scatter(
    u_centroide,
    v_centroide,
    color='blue',
    marker='x',
    s=100,
    label='Centroide'
)

ax.legend()
plt.show()

sigma = sqrt(autovalor_principal.real/autovalor_secundario.real)
print(f'Descritor de alongamento: {sigma:.2f}')
#-------------------------------------------------------------------------------------------------------------------------
# ETAPA 11: INFLUÊNCIA DA TRANSLAÇÃO

# Criar matriz de translação
deslocamento = np.float32([
    [1, 0, 40],
    [0, 1, -60]
])

# Deslocar o objeto sem alterar sua forma
altura, largura = mascara_objeto.shape

mascara_deslocada = cv2.warpAffine(
    mascara_objeto,
    deslocamento,
    (largura, altura),
    flags=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

# Mostrar objeto deslocado
plt.imshow(mascara_deslocada, cmap='gray')
plt.title('Objeto após translação')
plt.show()

# Calcular momento de ordem zero
m00_deslocado = np.sum(mascara_deslocada)

# Calcular momentos de primeira ordem
v, u = np.indices(mascara_deslocada.shape)

m10_deslocado = np.sum(u * mascara_deslocada)
m01_deslocado = np.sum(v * mascara_deslocada)

# Calcular centroide
uc_deslocado = m10_deslocado / m00_deslocado
vc_deslocado = m01_deslocado / m00_deslocado

# Calcular momentos centrais de segunda ordem
mi20_deslocado = np.sum(
    ((u - uc_deslocado)**2) * mascara_deslocada
)

mi02_deslocado = np.sum(
    ((v - vc_deslocado)**2) * mascara_deslocada
)

mi11_deslocado = np.sum(
    ((u - uc_deslocado) * (v - vc_deslocado)) * mascara_deslocada
)

# Construir matriz J
matriz_J_deslocada = np.array([
    [mi20_deslocado, mi11_deslocado],
    [mi11_deslocado, mi02_deslocado]
])

# Calcular autovalores e autovetores
autovalores_deslocados, autovetores_deslocados = np.linalg.eig(
    matriz_J_deslocada
)

# Ordenar autovalores do maior para o menor
ordem = np.argsort(autovalores_deslocados)[::-1]

autovalores_deslocados = autovalores_deslocados[ordem]
autovetores_deslocados = autovetores_deslocados[:, ordem]

# Determinar orientação principal
q_u_deslocado = autovetores_deslocados[0, 0].real
q_v_deslocado = autovetores_deslocados[1, 0].real

theta_deslocado = np.degrees(
    np.arctan2(q_v_deslocado, q_u_deslocado)
)

# Apresentar resultados
print("\n--- ETAPA 11: OBJETO DESLOCADO ---")

print(f"Momento m00: {m00_deslocado}")
print(f"Centroide X: {uc_deslocado:.2f}")
print(f"Centroide Y: {vc_deslocado:.2f}")

print("\nMomentos centrais:")
print(f"mi20: {mi20_deslocado}")
print(f"mi02: {mi02_deslocado}")
print(f"mi11: {mi11_deslocado}")

print("\nMatriz J:")
print(matriz_J_deslocada)

print("\nAutovalores:")
print(autovalores_deslocados)

print(f"\nOrientação: {theta_deslocado:.2f} graus")