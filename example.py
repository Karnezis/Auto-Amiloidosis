import autokeras as ak # Importa o AutoKeras
import tensorflow as tf # Importa o TensorFlow
import os # Importa o sistema para lidar com arquivos

local_dir_path = os.path.dirname(os.path.abspath(__file__)) # Pego o caminho absoluto do diretório
traindata_dir = os.path.join(local_dir_path, 'PS-Amiloidosis', 'train') # Crio o caminho de treino
valdata_dir = os.path.join(local_dir_path, 'PS-Amiloidosis', 'validation') # Crio o caminho de validação

# ---------------- PERSONALIZÁVEL ----------------
batch_size = 16 #32 # MUDAR # Tamanho do Batch
img_height = 244 # Altura da Imagem
img_width = 244 # Largura da Imagem

# Pega os dados de treino do diretório
train_data = ak.image_dataset_from_directory(
    traindata_dir, # Diretório de treino
    #validation_split=0.2, # Use 20% para validação
    # Define se é o dataset de treino ou de validação caso o split esteja definido
    #subset="training", 
    seed=123, # Dá um shuffle na ordem de pegar as imagens
    image_size=(img_height, img_width), # Redimensiona as imagens
    batch_size=batch_size) # Batch size né '-'

# Pega os dados de validação do diretório
test_data = ak.image_dataset_from_directory(
    valdata_dir, # Diretório de validação
    #validation_split=0.2, # Use 20% para validação
    # Define se é o dataset de treino ou de validação caso o split esteja definido
    #subset="validation",
    seed=123, # Dá um shuffle na ordem de pegar as imagens
    image_size=(img_height, img_width), # Redimensiona as imagens
    batch_size=batch_size) # Batch size né '-'

# Classe de classificação de imagem do AutoKeras
clf = ak.ImageClassifier(overwrite=True, # Sobrescreve um projeto existente com o mesmo nome se algum for encontrado.
    project_name='AutoAmiloidosis', # Nome do projeto (fica bonito)
    max_trials=1, # O número máximo de diferentes modelos Keras para tentar # MUDAR
    metrics=[tf.keras.metrics.Accuracy(), # Lista de Métricas a Serem Avaliadas
    tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.FalseNegatives()])

# Método de Treino
clf.fit(train_data, # Dados de Treino
    epochs=1) # Número inteiro de épocas # MUDAR

# Avaliação da Rede
arq = open("validation.txt","w+")
# Avalia o desempenho da rede no dataset de validação
nota = "[Loss, Accuracy, BinaryAccuracy, AUC, Precision, Recall, FalsePositives, FalseNegatives]" + str(clf.evaluate(test_data))
arq.write(nota)
arq.close()

# Exporta o melhor modelo em formato Keras
model = clf.export_model()
model.save("model_autoamiloidosis.h5")