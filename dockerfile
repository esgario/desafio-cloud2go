FROM python:3.10

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install -U pip setuptools wheel
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# Comando para rodar a aplicação com Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]