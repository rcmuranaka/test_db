import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image

# --- 1. Configuração e Carregamento ---
# Tenta carregar o modelo treinado
try:
    modelo_carregado = keras.models.load_model('classificador_roupas.h5')
except:
    st.error("Erro: O arquivo 'classificador_roupas.h5' não foi encontrado. Certifique-se de ter treinado e salvado o modelo primeiro.")
    st.stop()

# Definição das classes (deve ser a mesma usada no treinamento)
nomes_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- 2. Interface Streamlit ---
st.title('👕 Classificador de Imagens de Roupas com IA')
st.markdown('Faça o upload de uma imagem (28x28 pixels) de roupa para a IA prever o item.')

arquivo_upload = st.file_uploader("Escolha um arquivo de imagem...", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    # --- 3. Processamento da Imagem ---
    imagem = Image.open(arquivo_upload).convert('L') # Converte para escala de cinza
    
    # Redimensiona para o formato 28x28 que o modelo espera
    imagem_redimensionada = imagem.resize((28, 28))
    
    # Converte a imagem para um array numpy e normaliza
    imagem_array = np.array(imagem_redimensionada) / 255.0
    
    # Adiciona a dimensão 'batch' para o Keras (1 imagem, 28, 28)
    imagem_processada = (np.expand_dims(imagem_array, 0)) 

    # --- 4. Previsão (Inferência) ---
    st.image(imagem_redimensionada, caption='Imagem para Análise', width=100)
    st.write("Analisando...")
    
    # O modelo faz a previsão
    previsoes = modelo_carregado.predict(imagem_processada)
    
    # Encontra a classe com a maior probabilidade
    indice_previsao = np.argmax(previsoes[0])
    confianca = np.max(previsoes[0]) * 100

    # --- 5. Exibição do Resultado ---
    st.success(f"✅ Previsão da IA: **{nomes_classes[indice_previsao]}**")
    st.info(f"Confiança: **{confianca:.2f}%**")
