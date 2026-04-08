import uvicorn
from openenv.core.env_server.http_server import create_app
from env import LlamaRouterEnv

# Wrap our environment in the standard OpenEnv HTTP server
app = create_app(LlamaRouterEnv)

def main():
    # Hugging Face Spaces require port 7860
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()