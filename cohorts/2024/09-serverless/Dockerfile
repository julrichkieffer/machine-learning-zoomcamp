FROM agrigorev/model-2024-hairstyle:v3

COPY ["lambda_function.py", "constrains.txt", "./"]

RUN pip install --no-cache-dir -c constrains.txt pillow && \
    pip install --no-cache-dir -c constrains.txt https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl


CMD [ "lambda_function.lambda_handler" ]