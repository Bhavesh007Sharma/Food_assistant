# Use an official lightweight Python image.
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that Chainlit will use (default is 7860, but adjust if needed)
EXPOSE 7860

# Command to run the Chainlit app:
# This command starts the app with host 0.0.0.0 so it's accessible externally
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
