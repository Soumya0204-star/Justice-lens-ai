from dotenv import load_dotenv
import os

load_dotenv()
print("API Key:", os.getenv("WATSONX_API_KEY"))
print("Project ID:", os.getenv("WATSONX_PROJECT_ID"))
print("Current directory:", os.getcwd())