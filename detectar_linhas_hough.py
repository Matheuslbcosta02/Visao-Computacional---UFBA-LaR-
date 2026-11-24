import cv2
import matplotlib.pyplot as plt
import numpy as np

imagem_bgr = cv2.imread('imagem_detectar_linhas.webp')
imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
imagem_cinza = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY)

plt.subplot(1,2,1)
plt.imshow(imagem_rgb)
plt.subplot(1,2,2)
plt.imshow(imagem_cinza, cmap='gray')
plt.show()



print(f'Tipo de dado: {imagem_rgb.dtype}')
print(f'Altura, largura e canais: {imagem_rgb.shape}')
print(f'Total de pixels: {imagem_rgb.shape[0] * imagem_rgb.shape[1]}')


# Aplicar detector de bordas Canny
limiar_inferior = 100
limiar_superior = 200

imagem_bordas = cv2.Canny(imagem_cinza, limiar_inferior, limiar_superior)

# Visualizar imagem em tons de cinza e imagem de bordas
plt.subplot(1,2,1)
plt.imshow(imagem_cinza, cmap='gray')
plt.title('Imagem em tons de cinza')

plt.subplot(1,2,2)
plt.imshow(imagem_bordas, cmap='gray')
plt.title('Detecção de bordas - Canny')

plt.show()

from skimage.transform import hough_line , hough_line_peaks

# Aplicar Transformada de Hough

angulos = np.deg2rad(np.arange(-90, 90, 1))

acumulador, valores_angulos, valores_rho = hough_line(
    imagem_bordas,
    theta=angulos
)

print('Dimensões do acumulador:', acumulador.shape)

print('Intervalo de rho:')
print(valores_rho.min(), valores_rho.max())

print('Intervalo de theta:')
print(np.rad2deg(valores_angulos.min()),
      np.rad2deg(valores_angulos.max()))

plt.figure(figsize=(10, 6))

plt.imshow(
    acumulador,
    aspect='auto',
    cmap='hot',
    extent=[
        np.rad2deg(valores_angulos.min()),
        np.rad2deg(valores_angulos.max()),
        valores_rho.max(),
        valores_rho.min()
    ]
)

plt.xlabel('Theta (graus)')
plt.ylabel('Rho (pixels)')
plt.title('Matriz Acumuladora de Hough')
plt.colorbar(label='Número de votos')

plt.show()

acumulador_picos, angulos_picos, rho_picos = hough_line_peaks(
    acumulador,
    valores_angulos,
    valores_rho,
    num_peaks=5
)

# Apresentar os parâmetros encontrados
for i in range(len(acumulador_picos)):
    print(f'Pico {i+1}')
    print(f'Rho: {rho_picos[i]}')
    print(f'Theta: {np.rad2deg(angulos_picos[i]):.2f} graus')
    print(f'Número de votos: {acumulador_picos[i]}')
    print('----------------------')

    # Visualizar os picos no acumulador

plt.figure(figsize=(10, 6))

plt.imshow(
    acumulador,
    aspect='auto',
    cmap='hot',
    extent=[
        np.rad2deg(valores_angulos.min()),
        np.rad2deg(valores_angulos.max()),
        valores_rho.max(),
        valores_rho.min()
    ]
)

plt.scatter(
    np.rad2deg(angulos_picos),
    rho_picos,
    color='cyan',
    marker='x',
    s=100,
    label='Picos detectados'
)

plt.xlabel('Theta (graus)')
plt.ylabel('Rho (pixels)')
plt.title('Picos da Matriz Acumuladora de Hough')
plt.legend()
plt.colorbar(label='Número de votos')
plt.show()

imagem_linhas = imagem_rgb.copy()

for rho, theta in zip(rho_picos, angulos_picos):

    # Calcular ponto de referência da reta
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    # Calcular os pontos extremos da reta
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)

    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    # Desenhar reta na imagem
    cv2.line(
        imagem_linhas,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

# Visualizar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imagem_bordas, cmap='gray')
plt.title('Imagem de bordas')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(acumulador, aspect='auto', cmap='hot')
plt.title('Matriz acumuladora')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(imagem_linhas)
plt.title('Linhas detectadas')
plt.axis('off')

plt.tight_layout()
plt.show()

theta = np.rad2deg(angulos_picos[0])

# Calcular a orientação da própria reta
orientacao_linha = (theta + 90) % 180

print('Análise da orientação da linha')
print(f'Theta (normal da reta): {theta:.2f} graus')
print(f'Orientação da própria reta: {orientacao_linha:.2f} graus')


# Etapa 09 - Influência do limiar de votação

# Definir os três limiares de votação
limiares = [150, 200, 230]

# Criar figura para comparar os resultados
plt.figure(figsize=(18, 6))

for i, limiar in enumerate(limiares):

    # Identificar os picos considerando o limiar atual
    acumulador_picos, angulos_picos, rho_picos = hough_line_peaks(
        acumulador,
        valores_angulos,
        valores_rho,
        threshold=limiar,
        num_peaks=50
    )

    # Criar cópia da imagem original
    imagem_resultado = imagem_rgb.copy()

    # Desenhar as linhas encontradas
    for rho, theta in zip(rho_picos, angulos_picos):

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(
            imagem_resultado,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Apresentar informações no terminal
    print(f'\nLimiar de votação: {limiar}')
    print(f'Número de picos encontrados: {len(acumulador_picos)}')

    for j in range(len(acumulador_picos)):
        print(f'Linha {j+1}')
        print(f'Rho: {rho_picos[j]}')
        print(f'Theta: {np.rad2deg(angulos_picos[j]):.2f} graus')
        print(f'Votos: {acumulador_picos[j]}')

    # Mostrar imagem correspondente ao limiar
    plt.subplot(1, 3, i+1)
    plt.imshow(imagem_resultado)
    plt.title(f'Limiar de votação: {limiar}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Etapa 10 - Influência da resolução angular

# Definir as resoluções angulares
resolucoes = [0.5, 1, 2]

# Manter o mesmo limiar de votação nos experimentos
limiar_votacao = 150

plt.figure(figsize=(18, 6))

for i, resolucao in enumerate(resolucoes):

    # Criar os ângulos com a resolução atual
    angulos = np.deg2rad(
        np.arange(-90, 90, resolucao)
    )

    # Aplicar Transformada de Hough
    acumulador_res, valores_angulos_res, valores_rho_res = hough_line(
        imagem_bordas,
        theta=angulos
    )

    # Identificar os picos
    picos, angulos_picos_res, rho_picos_res = hough_line_peaks(
        acumulador_res,
        valores_angulos_res,
        valores_rho_res,
        threshold=limiar_votacao,
        num_peaks=5
    )

    # Criar cópia da imagem original
    imagem_resultado = imagem_rgb.copy()

    # Desenhar as linhas encontradas
    for rho, theta in zip(rho_picos_res, angulos_picos_res):

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(
            imagem_resultado,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Apresentar informações no terminal
    print(f'\nResolução angular: {resolucao} graus')
    print(f'Dimensões do acumulador: {acumulador_res.shape}')

    print('Intervalo de theta:')
    print(
        np.rad2deg(valores_angulos_res.min()),
        np.rad2deg(valores_angulos_res.max())
    )

    print(f'Quantidade de picos encontrados: {len(picos)}')

    for j in range(len(picos)):
        print(f'\nLinha {j+1}')
        print(f'Rho: {rho_picos_res[j]}')
        print(
            f'Theta: {np.rad2deg(angulos_picos_res[j]):.2f} graus'
        )
        print(f'Número de votos: {picos[j]}')

    # Visualizar resultado
    plt.subplot(1, 3, i+1)
    plt.imshow(imagem_resultado)
    plt.title(f'Resolução angular: {resolucao}°')
    plt.axis('off')

plt.tight_layout()
plt.show()



# Etapa 11 - Influência da imagem de bordas

# Definir diferentes configurações do Canny
configuracoes = [
    (50, 100),
    (150, 250)
]

# Manter a mesma resolução angular nos experimentos
angulos = np.deg2rad(np.arange(-90, 90, 1))

# Manter o mesmo limiar de votação para comparação
limiar_votacao = 150

# Criar figura para apresentar os resultados
plt.figure(figsize=(18, 10))

for i, (limiar_inf, limiar_sup) in enumerate(configuracoes):

    # Aplicar Canny com os parâmetros atuais
    imagem_bordas_teste = cv2.Canny(
        imagem_cinza,
        limiar_inf,
        limiar_sup
    )

    # Aplicar Transformada de Hough
    acumulador_teste, angulos_teste, rho_teste = hough_line(
        imagem_bordas_teste,
        theta=angulos
    )

    # Identificar os picos
    picos, angulos_picos_teste, rho_picos_teste = hough_line_peaks(
        acumulador_teste,
        angulos_teste,
        rho_teste,
        threshold=limiar_votacao,
        num_peaks=20
    )

    # Criar cópia da imagem original
    imagem_resultado = imagem_rgb.copy()

    # Desenhar as linhas encontradas
    for rho, theta in zip(rho_picos_teste, angulos_picos_teste):

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(
            imagem_resultado,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Calcular quantidade de pixels de borda
    quantidade_bordas = np.count_nonzero(imagem_bordas_teste)

    # Apresentar informações no terminal
    print(f'\nConfiguração Canny: {limiar_inf} / {limiar_sup}')

    print(f'Quantidade de pixels de borda: {quantidade_bordas}')

    print(f'Valor máximo do acumulador: {acumulador_teste.max()}')

    print(f'Número de picos encontrados: {len(picos)}')

    for j in range(len(picos)):
        print(f'\nLinha {j+1}')
        print(f'Rho: {rho_picos_teste[j]}')
        print(
            f'Theta: {np.rad2deg(angulos_picos_teste[j]):.2f} graus'
        )
        print(f'Número de votos: {picos[j]}')

    # Primeira coluna: imagem de bordas
    plt.subplot(2, 3, i*3 + 1)
    plt.imshow(imagem_bordas_teste, cmap='gray')
    plt.title(f'Canny: {limiar_inf}/{limiar_sup}')
    plt.axis('off')

    # Segunda coluna: matriz acumuladora
    plt.subplot(2, 3, i*3 + 2)
    plt.imshow(acumulador_teste, aspect='auto', cmap='hot')
    plt.title(f'Acumulador: {limiar_inf}/{limiar_sup}')
    plt.xlabel('Theta')
    plt.ylabel('Rho')

    # Terceira coluna: linhas detectadas
    plt.subplot(2, 3, i*3 + 3)
    plt.imshow(imagem_resultado)
    plt.title(f'Linhas: {limiar_inf}/{limiar_sup}')
    plt.axis('off')

plt.tight_layout()
plt.show()



# Etapa 12 
picos_etapa12, angulos_etapa12, rho_etapa12 = hough_line_peaks(
    acumulador,
    valores_angulos,
    valores_rho,
    num_peaks=5
)

# Selecionar a linha com maior quantidade de votos
rho = rho_etapa12[0]
theta = angulos_etapa12[0]
votos = picos_etapa12[0]

# Converter theta para graus
theta_graus = np.rad2deg(theta)

# Calcular a orientação da própria linha
orientacao_linha = (theta_graus + 90) % 180

# Criar cópia da imagem original
imagem_linha = imagem_rgb.copy()

# Calcular os pontos da reta
a = np.cos(theta)
b = np.sin(theta)

x0 = a * rho
y0 = b * rho

x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * a)

x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * a)

# Desenhar somente a linha selecionada
cv2.line(
    imagem_linha,
    (x1, y1),
    (x2, y2),
    (255, 0, 0),
    2
)

# Apresentar os parâmetros encontrados
print('\nEtapa 12 - Análise detalhada da linha')
print(f'Rho: {rho}')
print(f'Theta: {theta_graus:.2f} graus')
print(f'Número de votos: {votos}')
print(f'Orientação da linha: {orientacao_linha:.2f} graus')

# Visualizar a linha isoladamente na imagem original
plt.figure(figsize=(10, 7))

plt.imshow(imagem_linha)
plt.title('Análise detalhada - Linha selecionada')
plt.axis('off')

plt.show()
