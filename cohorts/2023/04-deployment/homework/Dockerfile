FROM svizor/zoomcamp-model:mlops-3.10.0-slim

RUN pip install pipenv && pipenv install scikit-learn==1.2.2 pandas pyarrow 

COPY starter.py ./
CMD ["pipenv", "run", "python", "starter.py", "-y", "2022", "-m", "4"]
