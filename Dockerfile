FROM pytorch/pytorch:latest

WORKDIR /app
COPY . .

RUN apt-get update
# RUN apt-get install git --assume-yes
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
# RUN python3 -m spacy download en_core_web_sm
# RUN python3 -m spacy download en_core_web_trf