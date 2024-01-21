FROM public.ecr.aws/lambda/python:3.10

RUN pip install requests numpy pillow tensorflow flask

COPY final-model.keras .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
