# FROM python:3.9
# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH
# WORKDIR $HOME/app
# COPY --chown=user . $HOME/app
# COPY ./requirements.txt ~/app/requirements.txt

# RUN pip install --prefer-binary --no-cache-dir -r requirements.txt && pip install chainlit
# COPY . .
# CMD ["chainlit", "run", "rag_app.py", "--port", "7860"]






# # Use the official slim Python 3.9 image for a smaller footprint
# FROM python:3.9-slim

# # Install any necessary build dependencies (optional)
# RUN apt-get update && apt-get install -y build-essential python3-dev && apt-get clean

# # Create a new user with a specific UID
# RUN useradd -m -u 1000 user

# # Set environment variables
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH

# # Set the working directory
# WORKDIR $HOME/app

# # Copy only the requirements file first to leverage Docker's layer caching
# COPY --chown=user requirements.txt $HOME/app/requirements.txt

# # Install dependencies, preferring binary packages where available to speed up builds
# RUN pip install --prefer-binary --no-cache-dir -r requirements.txt && pip install chainlit

# # Copy the rest of the application files into the container
# COPY --chown=user . $HOME/app

# # Expose the port that Chainlit will run on (optional, good for documentation)
# EXPOSE 7860

# # Run Chainlit with the specified command
# CMD ["chainlit", "run", "rag_app.py", "--port", "7860"]
# #CMD ["~/.local/bin/chainlit", "run", "rag_app.py", "--port", "7860"]

# # Use the official slim Python 3.9 image
# FROM python:3.9-slim

# # Install necessary build dependencies
# RUN apt-get update && apt-get install -y build-essential python3-dev && apt-get clean

# # Create a new user with a specific UID
# RUN useradd -m -u 1000 user

# # Set environment variables
# ENV HOME=/home/user \
#     PATH=/usr/local/bin:$PATH

# # Set the working directory
# WORKDIR $HOME/app

# # Copy only the requirements file first to leverage Docker's layer caching
# COPY --chown=user requirements.txt $HOME/app/requirements.txt

# # Install dependencies globally, including chainlit
# RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
#     && pip install --no-cache-dir chainlit

# # Verify chainlit installation and check global binaries
# RUN ls /usr/local/bin && pip show chainlit

# # Copy the rest of the application files into the container
# COPY --chown=user . $HOME/app

# # Expose the port that Chainlit will run on
# EXPOSE 7860

# # Run Chainlit with the specified command
# CMD ["chainlit", "run", "rag_app.py", "--port", "7860"]


# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install chainlit
RUN pip install --no-cache-dir chainlit

# Verify installations
RUN pip list

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV OPENAI_API_KEY=your_api_key_here

# Run app.py when the container launches
CMD ["python", "-m", "chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0"]