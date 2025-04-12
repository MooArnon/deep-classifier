# Use the Python 3.9 slim base image
FROM public.ecr.aws/lambda/python:3.9

# Install build dependencies
RUN yum update -y && yum install -y git libgomp

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install numpy, scipy, and scikit-learn
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt --timeout 600

# Copy your application code to the container
COPY deep_classifier ${LAMBDA_TASK_ROOT}/deep_classifier
COPY sql ${LAMBDA_TASK_ROOT}/sql
COPY *.py ${LAMBDA_TASK_ROOT}

COPY btc_fine_tuned_model_wrapper.pkl ${LAMBDA_TASK_ROOT}
COPY btc_fine_tuned_model.h5 ${LAMBDA_TASK_ROOT}
