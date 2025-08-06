# Mochammad Daffa Putra Karyudi

---

# Streamlit Application Tutorial

This repository contains a Streamlit app, and this guide will walk you through the steps to set it up and run it using `streamlit run`.

## Prerequisites

Before running the app, ensure that you have the following prerequisites:

1. **Python 3.10**  
   Make sure that you have Python 3.10 or later installed. You can check your Python version by running:
   ```bash
   python --version
   ```

2. **Install Dependencies**  
   This project uses `uv` for faster and more reliable Python package management. If you haven't installed `uv` yet, install it first:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then install dependencies from `requirements.txt`:
   ```bash
   uv pip install -r requirements.txt
   ```

   This will install `Streamlit` and other required packages using `uv`'s optimized installer.

## Running the App

Follow these steps to run the app:

### 1. Clone the Repository (If Applicable)

If you haven't already cloned the repository, do so with:
```bash
git clone 
cd ai-demo
```

### 2. Install the Dependencies

Ensure all required libraries are installed by running:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

In the terminal, navigate to the directory containing `app.py`. Then, run the following command to start the Streamlit application:
```bash
streamlit run app.py
```

### 4. Open the App in Your Browser

After running the above command, Streamlit will automatically open a new tab in your default web browser to show the application. If it doesn't open automatically, you can manually navigate to:
```
http://localhost:8501
```

### 5. Interact with the App

Once the app is running, you can interact with it directly through the web interface. Depending on the design of the app, you may be able to upload data, visualize results, and change parameters interactively.

### 6. Stop the App

To stop the app, return to the terminal and press `Ctrl+C`.

## Deployment Options

### 1. Local Development
For local development, follow the steps in the "Running the App" section above.

### 2. Docker Deployment
This project includes Docker support for easy deployment. Here are the steps to deploy using Docker:

#### Using Docker Directly
```bash
# Build the Docker image
docker build -t ai-demo .

# Run the container
docker run -p 8501:8501 ai-demo
```

#### Using Docker Compose (Recommended)
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The application will be available at `http://localhost:8501`.

### 3. Cloud Deployment Options

The application can be deployed to various cloud platforms:

1. **Streamlit Cloud** (Easiest)
   - Connect your GitHub repository to [Streamlit Cloud](https://streamlit.io/cloud)
   - Select your repository and branch
   - The app will be automatically deployed

2. **AWS**
   - Deploy using AWS Elastic Beanstalk
   - Use EC2 with Docker
   - Use ECS/EKS for container orchestration

3. **Google Cloud Platform**
   - Deploy using Google App Engine
   - Use Google Cloud Run for containerized deployment
   - Use GKE for Kubernetes deployment

4. **Azure**
   - Deploy using Azure App Service
   - Use Azure Container Instances
   - Use AKS for Kubernetes deployment

For detailed deployment instructions for each platform, refer to the [Streamlit Deployment Guide](https://discuss.streamlit.io/t/streamlit-deployment-guide-wiki/5099).

## Troubleshooting

### App Not Opening Automatically
If the app doesn't open in the browser, check for error messages in the terminal and ensure the correct port (`8501` by default) is not being blocked by a firewall.

### Docker Issues
1. If you get a port conflict, change the port mapping in docker-compose.yml
2. If the container exits immediately, check the logs using `docker-compose logs`
3. For permission issues, ensure the user has access to Docker daemon

### Dependency Issues
If you face any dependency issues, make sure that your environment is properly set up. You can try creating a fresh virtual environment with uv:
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
uv pip install -r requirements.txt
```

If you encounter any issues with uv:
1. Make sure you have the latest version: `uv --version`
2. Try updating uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Check if your Python version is compatible
4. For Windows users, use PowerShell or Git Bash for installation