
# # WORKS
# # Use the official slim Python 3.9 image
# FROM python:3.9-slim

# # Install necessary build dependencies
# RUN apt-get update && apt-get install -y build-essential python3-dev libffi-dev && apt-get clean

# # Create a new user with a specific UID
# RUN useradd -m -u 1000 user

# # Set environment variables
# ENV HOME=/home/user \
#     PATH=/usr/local/bin:$PATH

# # Set the working directory
# WORKDIR $HOME/app

# # Copy only the requirements file first to leverage Docker's layer caching
# COPY --chown=user requirements.txt $HOME/app/requirements.txt

# # Install dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir protobuf==3.20.0 && \
#     pip install --no-cache-dir -r requirements.txt

# # Install chainlit separately
# RUN pip install --no-cache-dir chainlit

# # Verify chainlit installation and check global binaries
# RUN ls /usr/local/bin && pip show chainlit

# # Copy the rest of the application files into the container
# COPY --chown=user . $HOME/app

# # Expose the port that Chainlit will run on
# EXPOSE 7860

# # Run Chainlit with the specified command
# CMD ["chainlit", "run", "app.py", "--port", "7860"]


# Use the official slim Python 3.9 image
FROM python:3.9-slim

# Install necessary build dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev libffi-dev && apt-get clean

# Create a new user with a specific UID
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy only the requirements file first to leverage Docker's layer caching
COPY --chown=user requirements.txt $HOME/app/requirements.txt

# Print requirements.txt content for debugging
RUN cat requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install chainlit explicitly and verify its installation
RUN pip install --no-cache-dir chainlit==0.7.700 && \
    which chainlit && \
    chainlit --version

# Verify all installations
RUN pip list

# Copy the rest of the application files into the container
COPY --chown=user . $HOME/app

# Expose the port that Chainlit will run on
EXPOSE 7860

# Switch to the non-root user
USER user

# Run Chainlit with the specified command
CMD ["chainlit", "run", "app.py", "--port", "7860"]