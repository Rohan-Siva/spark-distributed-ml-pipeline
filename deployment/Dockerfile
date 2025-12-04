FROM python:3.9-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean;

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/

# Set entrypoint
ENTRYPOINT ["python", "src/pipeline.py"]
