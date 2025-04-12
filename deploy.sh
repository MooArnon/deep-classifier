#!/bin/bash

# Variables
REPOSITORY_NAME="space-time/deep-classifier"
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}
REGION="ap-southeast-1"

# Get the latest image tag from ECR
latest_tag=$(aws ecr describe-images --repository-name $REPOSITORY_NAME \
  --query 'sort_by(imageDetails,& imagePushedAt)[-1].imageTags[0]' --output text --region $REGION)


# Extract the numeric part of the tag and increment it
if [[ $latest_tag =~ ^([0-9]+)$ ]]; then
  new_tag=$(($latest_tag + 1))
else
  new_tag=1
fi

aws s3 cp s3://space-time-model/classifier/tuned/btc/dnn/ . --recursive

# Build the Docker image with the new tag
sudo docker build --platform linux/arm64 -t $REPOSITORY_NAME:$new_tag .

# Tag the Docker image for ECR
docker tag $REPOSITORY_NAME:$new_tag $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$new_tag

# Log in to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Push the image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$new_tag

echo "Image pushed with tag: $new_tag"