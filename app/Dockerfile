FROM python:3.8
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python3","app_gradio.py"]

