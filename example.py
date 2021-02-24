import autokeras as ak # Importa o AutoKeras
import tensorflow as tf # Importa o TensorFlow
import os # Importa o sistema para lidar com arquivos

local_dir_path = os.path.dirname(os.path.abspath(__file__)) # Pego o caminho absoluto do diretório
traindata_dir = os.path.join(local_dir_path, 'PS-Amiloidosis', 'train') # Crio o caminho de treino
valdata_dir = os.path.join(local_dir_path, 'PS-Amiloidosis', 'validation') # Crio o caminho de validação

# ---------------- PERSONALIZÁVEL ----------------
batch_size = 32 # Tamanho do Batch
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

clf = ak.ImageClassifier(overwrite=True, # Sobrescreve um projeto existente com o mesmo nome se algum for encontrado.
    project_name='AutoAmiloidosis', # Nome do projeto (fica bonito)
    max_trials=1) # O número máximo de diferentes modelos Keras para tentar
clf.fit(train_data, epochs=1)
print(clf.evaluate(test_data))

# Export as a Keras Model.
model = clf.export_model()
model.save("model_autoamiloidosis.h5")