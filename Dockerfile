FROM tensorflow/tensorflow:2.0.0a0

RUN pip install sagemaker-containers

# Copies the training code inside the container
COPY BookSterMatrixFactorization.py /opt/ml/code/BookSterMatrixFactorization.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM BookSterMatrixFactorization.py