# Execução com Docker

## Pré-requisitos
* Instalar o `Docker`.

[Instalação oficial](https://docs.docker.com/get-docker/)
[Instalação no Fedora](https://computingforgeeks.com/how-to-install-docker-on-fedora/)

## Construindo a imagem
A imagem para treinamento é o arquivo `Dockerfile`.

Execute o comando abaixo para gerar a imagem a ser utilizada pelo container.

``` bash
docker build -t ps-amiloidosis .
```

## Levantando o ambiente
* Criar o container a partir da imagem `ps-amiloidosis` gerada anteriormente executando o comando abaixo.

``` bash
docker run --gpus all -it -v {{base-path}}/Auto-Amiloidosis:/usr/src/app -w /usr/src/app ps-amiloidosis bash
```

Dessa forma o terminal do container estara integrado com o terminal do host permitindo a execucao de comandos.

## Rodar o script no container

Estando no terminal integrado do container com o host execute o comando
``` bash
python example.py
```

## Copiar Arquivos do Docker para o Host

``` bash
sudo docker cp {{Image ID}}:/usr/src/app/validation.txt {{Absolute Path}}
```