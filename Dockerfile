# First stage: Build
FROM python:3.12-slim AS build

# Set the working directory
WORKDIR /app

# Install build tools and clean up afterwards
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Upgrade pip separately
RUN python -m pip install --upgrade pip

# Install dependencies - use --no-warn-script-location to suppress warnings
RUN pip install --no-cache-dir --no-warn-script-location -r requirements.txt

# Second stage: Final image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Create a non-privileged user
RUN adduser --disabled-password --gecos "" appuser

# Copy only the installed dependencies from the build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy the entire app directory
COPY . .

# Change ownership of the application files to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Command to run the Streamlit app
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
