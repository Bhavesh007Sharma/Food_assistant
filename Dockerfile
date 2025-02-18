FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV HF_HOME=/app/.cache

RUN chmod -R 777 /app

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
