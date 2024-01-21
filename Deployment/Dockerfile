FROM public.ecr.aws/lambda/python:3.10

RUN pip install requests numpy pillow tensorflow

COPY final-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
