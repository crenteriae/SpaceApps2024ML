FROM python:3.11.5-slim

WORKDIR /app

COPY model test_data.csv preds.py server.py requirements.txt /app/

RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "server:app", "-b", "0.0.0.0:5000", "-w", "4"]