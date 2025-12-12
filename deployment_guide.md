# Deployment Guide for GCP

This guide will help you set up your FastAPI server on a fresh Google Cloud Platform (GCP) Compute Engine instance.

## 1. Instance Setup
- **OS**: Ubuntu 22.04 LTS (Recommended) or Debian
- **Firewall**: configuration is crucial.
    - Go to **VPC Network** > **Firewall**.
    - Create a rule to **allow ingress** on tcp port `8000`.
    - Source filters: `0.0.0.0/0` (Allows access from anywhere).

## 2. Server Configuration
Once you SSH into your instance, run the following commands:

### Update System and Install Python/Pip
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git
```

### Clone Your Repository
(Or upload your files using scp/gcloud)
```bash
git clone <YOUR_REPO_URL>
cd disease-detection-api
```

### Setup Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Running the Server

### For Testing (Foreground)
This will stop when you close the terminal.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### For Production (Background Service)
Use `nohup` to keep it running after you disconnect:

```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
```

- To stop it later: `pkill uvicorn`
- To view logs: `tail -f app.log`

## 4. Connecting from Flutter
- **Base URL**: `http://<YOUR_EXTERNAL_IP>:8000`
- **Endpoint**: `/infer` or `/demo/infer`

## Notes on 0.0.0.0
- **Is it safe?** Yes, for a public API. `0.0.0.0` tells the server to listen on all network interfaces (internal requests, external web requests). Without this, your GCP instance would only listen to itself (`localhost`), and you couldn't access it from the internet.
- **Security**: Since you are allowing all origins (`*`) and listening on all interfaces, anyone with the IP can access it. For a production health app, you would eventually want to add API Key authentication or restrict CORS to your specific app domain.
