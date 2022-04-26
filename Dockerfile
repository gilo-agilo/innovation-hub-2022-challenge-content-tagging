FROM python:3.7.13

RUN mkdir app
WORKDIR /app
COPY . .

RUN python -m pip install --trusted-host pypi.org -v --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt -t .

CMD ["python", "app.py"]