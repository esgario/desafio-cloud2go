# desafio-cloud2go


## Executando a API

Para executar a API, você precisa ter o Docker instalado. O primeiro passo é construir a imagem do Docker:

```
docker build -t cloud2go-app .
```

Depois de construir a imagem, você pode executar o contêiner. O comando abaixo irá mapear a porta 8000 do contêiner para a porta 8000 da sua máquina local:
```
docker run -p 8000:8000 cloud2go-app
```

## Realizando inferência

Em outro terminal rode o script de inferência:

```
python inference.py
```
