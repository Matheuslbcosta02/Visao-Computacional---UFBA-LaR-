import cv2
import matplotlib.pyplot as plt
import numpy as np


imagem_bgr = cv2.imread('imagem_classificar_cores.jpg')
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(imagem_rgb)
plt.show()

print(f'Tipo de dado: {imagem_rgb.dtype}')
print(f'Altura, largura e canais: {imagem_rgb.shape}')
print(f'Total de pixels: {imagem_rgb.shape[0] * imagem_rgb.shape[1]}')



imagem_croma = imagem_rgb.astype(np.float32)

R = imagem_croma[:, :, 0]
G = imagem_croma[:, :, 1]
B = imagem_croma[:, :, 2]

Y = R + G + B

Y_sem_zero = np.where(Y == 0, 0.000001, Y)

r = R / Y_sem_zero
g = G / Y_sem_zero
b = B / Y_sem_zero

print(r + g + b)


plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.imshow(r, cmap='gray')
plt.title('r')

plt.subplot(1, 2, 2)
plt.imshow(g, cmap='gray')
plt.title('g')

plt.show()



r_amostra = r[::20, ::20].flatten()
g_amostra = g[::20, ::20].flatten()



coords_objeto = [
    (300, 170),
    (310, 210),
    (290, 190)
]

r_objeto = [r[v, u] for v, u in coords_objeto]
g_objeto = [g[v, u] for v, u in coords_objeto]



plt.scatter(
    r_amostra,
    g_amostra,
    s=5,
    color='red',
    alpha=0.8,
    label='pixels da imagem'
)

plt.scatter(
    0.333,
    0.333,
    s=70,
    marker='x',
    color='blue',
    alpha=1.0,
    label='branco ideal'
)

plt.scatter(
    r_objeto,
    g_objeto,
    s=120,
    marker='*',
    color='yellow',
    label='objeto'
)

plt.title('Plano de cromaticidade')
plt.legend()
plt.show()



rmin, rmax = 0.50, 0.70
gmin, gmax = 0.10, 0.30


M_C = (
    (r >= rmin) & (r <= rmax) &
    (g >= gmin) & (g <= gmax)
).astype(np.uint8)


regiao_M_C = cv2.bitwise_and(
    imagem_rgb,
    imagem_rgb,
    mask=M_C
)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(M_C, cmap='gray')
plt.title('M_C (Máscara Cromaticidade)')

plt.subplot(1, 2, 2)
plt.imshow(regiao_M_C)
plt.title('Região Segmentada M_C')

plt.show()

X = imagem_rgb.reshape(-1,3)
print(X.shape)

X = X.astype(np.float32)

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=3,
    max_iter=50,
    algorithm='elkan'
)

labels = kmeans.fit_predict(X)
centroides = kmeans.cluster_centers_

print("Centróides RGB:")
print(centroides)
imagem_kmeans = centroides[labels].reshape(imagem_rgb.shape).astype(np.uint8)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(imagem_kmeans)
plt.title("Imagem K-means (k=3)")

plt.show()


resultados = {
    3: (imagem_kmeans, centroides)
}

# Testar os outros valores de k
for k in [2, 5]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=3,
        max_iter=50,
        algorithm='elkan'
    )

    labels = kmeans.fit_predict(X)
    centroides_k = kmeans.cluster_centers_

    imagem_k = centroides_k[labels].reshape(
        imagem_rgb.shape
    ).astype(np.uint8)

    resultados[k] = (imagem_k, centroides_k)

    print(f"\nCentróides RGB para k={k}:")
    print(centroides_k)


# Apresentar os três resultados na mesma figura
plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem Original")
plt.axis("off")

for posicao, k in enumerate([2, 3, 5], start=2):

    imagem_resultado = resultados[k][0]

    plt.subplot(1, 4, posicao)
    plt.imshow(imagem_resultado)
    plt.title(f"K-means (k={k})")
    plt.axis("off")

plt.tight_layout()
plt.show()

SSE_final = kmeans.inertia_

print(f"SSE final para k=5: {SSE_final:.2f}")


# Etapa 10 - Seleção do cluster de interesse

# Selecionar o cluster das maçãs vermelhas
cluster_interesse = 0

# Recuperar os centróides de k=5
centroides5 = resultados[5][1]

# Organizar as etiquetas dos pixels na dimensão da imagem
labels_imagem = labels.reshape(imagem_rgb.shape[:2])

# Criar a máscara binária MK
M_K = (labels_imagem == cluster_interesse).astype(np.uint8)

# Selecionar a região correspondente ao cluster
regiao_M_K = cv2.bitwise_and(
    imagem_rgb,
    imagem_rgb,
    mask=M_K
)

# Mostrar o centróide selecionado
centroide_selecionado = centroides5[cluster_interesse]

print("Cluster selecionado:", cluster_interesse)
print("Centróide RGB:", centroide_selecionado)

# Criar uma imagem representando a cor do centróide
cor_cluster = np.full(
    (100, 100, 3),
    centroide_selecionado
).astype(np.uint8)

# Apresentar os resultados
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(resultados[5][0])
plt.title("Imagem com todos os clusters")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(cor_cluster)
plt.title("Centróide selecionado")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(M_K, cmap="gray")
plt.title("Máscara M_K")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(regiao_M_K)
plt.title("Região selecionada")
plt.axis("off")

plt.tight_layout()
plt.show()


# Etapa 11 - Comparação entre os métodos

# Transformar as máscaras em valores booleanos
MC_bool = M_C.astype(bool)
MK_bool = M_K.astype(bool)

# Calcular interseção e união
intersecao = np.logical_and(MC_bool, MK_bool)
uniao = np.logical_or(MC_bool, MK_bool)

# Quantidade de pixels selecionados
NC = np.sum(MC_bool)
NK = np.sum(MK_bool)

# Quantidade de pixels na interseção e união
N_intersecao = np.sum(intersecao)
N_uniao = np.sum(uniao)

# Calcular o índice de Jaccard
J = N_intersecao / N_uniao if N_uniao > 0 else 0

# Mostrar os resultados numéricos
print("\nComparação entre os métodos:")
print("Pixels selecionados por cromaticidade (NC):", NC)
print("Pixels selecionados pelo K-means (NK):", NK)
print("Pixels da interseção:", N_intersecao)
print("Pixels da união:", N_uniao)
print(f"Índice de Jaccard (IoU): {J:.4f}")

# Regiões selecionadas por cada método
regiao_MC = cv2.bitwise_and(
    imagem_rgb, imagem_rgb, mask=M_C
)

regiao_MK = cv2.bitwise_and(
    imagem_rgb, imagem_rgb, mask=M_K
)

# Apresentar comparação visual
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(imagem_rgb)
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(M_C, cmap="gray")
plt.title("Máscara Cromaticidade (MC)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(M_K, cmap="gray")
plt.title("Máscara K-means (MK)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(regiao_MC)
plt.title("Região selecionada por MC")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(regiao_MK)
plt.title("Região selecionada por MK")
plt.axis("off")

plt.tight_layout()
plt.show()
